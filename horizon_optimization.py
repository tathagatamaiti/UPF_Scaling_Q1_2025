import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, NonNegativeIntegers,
    Objective, Constraint, ConstraintList, SolverFactory, maximize
)
import math

def local_start_time(pdu_id, pdu_chunk_dict, chunk_start):
    """
    Returns the local start time for a given PDU session within the current chunk.

    The "local start time" is defined as the maximum between:
      - The actual session start time
      - The beginning of the current chunk

    Parameters
    ----------
    pdu_id : int
        The unique identifier of the PDU session.
    pdu_chunk_dict : dict
        A dictionary derived from the PDU dataframe with 'start', 'end', etc. keyed by PDU ID.
    chunk_start : int
        The start time of the current chunk we are optimizing over.

    Returns
    -------
    int
        The local (effective) start time of the PDU session in the current chunk.
    """
    real_start = pdu_chunk_dict['start'][pdu_id]
    return max(chunk_start, real_start)

def local_activity_duration(pdu_id, pdu_chunk_dict, chunk_start, chunk_end):
    """
    Returns the activity duration of a given PDU session within the current chunk.

    The "local activity duration" is the length of time for which the session is
    actually active in the current chunk. Specifically, it is the difference between
    the local effective end time and local start time, where:
      - The local start time is at least the chunk start
      - The local end time is at most the chunk end

    Parameters
    ----------
    pdu_id : int
        The unique identifier of the PDU session.
    pdu_chunk_dict : dict
        A dictionary derived from the PDU dataframe with 'start', 'end', etc. keyed by PDU ID.
    chunk_start : int
        The start time of the current chunk we are optimizing over.
    chunk_end : int
        The end time of the current chunk we are optimizing over.

    Returns
    -------
    int
        The duration (in time units) that the PDU is active within the chunk.
        Could be zero if the PDU session is not active during this chunk.
    """
    real_end = pdu_chunk_dict['end'][pdu_id]
    real_start = pdu_chunk_dict['start'][pdu_id]
    eff_start = max(chunk_start, real_start)
    eff_end   = min(chunk_end, real_end)
    return eff_end - eff_start

# =======================================================================
# Main simulation parameters
# =======================================================================
chunk_size = 5        # The time-window size (in same time units as the PDU sessions).
alpha = 0.8           # CPU capacity fraction that can be allocated for sessions.
beta = 1.0            # Weighting factor for dynamic energy consumption.
gamma = 0.01          # Weighting factor for minimizing the number of active UPFs.
delta = 0.001         # Weighting factor penalizing the deviation from minimum latency.
kappa = 0.001         # Weighting factor penalizing overall energy usage.
M = 1e9               # A large constant used for Big-M constraints in linear formulations.

# =======================================================================
# Load data
# =======================================================================
pdu_data = pd.read_csv('pdu_sessions_3600_42_0.5.csv')
upf_data = pd.read_csv('upf_instances_100_42.csv')

# Determine global start and end times across all PDUs
min_time = pdu_data['start'].min()
max_time = pdu_data['end'].max()

min_time_int = int(math.floor(min_time))
max_time_int = int(math.ceil(max_time))
chunk_size_int = int(chunk_size)

# Create chunk boundaries from the minimum to the maximum time.
chunk_boundaries = range(min_time_int, max_time_int + 1, chunk_size_int)

# Keep track of which UPF each PDU is assigned to (for the no-migration constraint).
previous_assignments = {}

# Lists for storing the solution details and system power usage over time.
solution_records = []
system_power_records = []

# Set of PDUs that we have already tried optimizing in a chunk.
previously_optimized_pdus = set()

# Create a solver factory for SCIP, specifying the path to the executable.
solver = SolverFactory('scip', executable='/home/tmaiti/Downloads/SCIPOptSuite-9.1.1-Linux/bin/scip')

# =======================================================================
# Main chunked horizon approach
# =======================================================================
for chunk_start in chunk_boundaries:
    """
    For each chunk boundary in the time horizon:
      1. Define the chunk_end as chunk_start + chunk_size - 1.
      2. Filter out PDUs that are active in this chunk but have not been solved before.
      3. Build the model, define sets, parameters, decision variables, objective, constraints.
      4. Solve the model with the chosen solver.
      5. Post-process results to record solution details and track partial sessions.
    """
    chunk_end = chunk_start + chunk_size - 1

    # Identify PDUs active in the chunk and not previously optimized
    active_mask = (
        (pdu_data['start'] <= chunk_end) &
        (pdu_data['end'] >= chunk_start) &
        (~pdu_data['id'].isin(previously_optimized_pdus))
    )
    pdu_chunk_df = pdu_data[active_mask].copy()

    # If no new PDUs appear in this chunk, skip solving this chunk to save time.
    if pdu_chunk_df.empty:
        print(f"No new PDUs in chunk {chunk_start} to {chunk_end}, skipping solve.")
        continue

    # Create a concrete model
    model = ConcreteModel()

    # ===================================================================
    # Sets
    # ===================================================================
    # Time set for this chunk
    time_points_chunk = list(range(chunk_start, chunk_end + 1))
    model.T = Set(initialize=time_points_chunk, ordered=True)

    # Set of PDUs active in this chunk
    model.A = Set(initialize=pdu_chunk_df['id'].tolist())

    # Set of UPF instances
    model.U = Set(initialize=upf_data['instance_id'].tolist())

    # Create dictionary versions of data for easier referencing in constraints
    pdu_chunk_dict = pdu_chunk_df.set_index('id').to_dict()
    upf_dict = upf_data.set_index('instance_id').to_dict()

    # ===================================================================
    # Helper Functions
    # ===================================================================
    def tmax_init(m, j):
        """
        Returns the maximum allowed latency for PDU session j.

        Used for bounding possible CPU-related intermediate constraints.
        """
        return m.l_high[j]

    def smax_init(m, i):
        """
        Defines the maximum CPU share per time-step that can be allocated on UPF i.

        This is typically alpha*C[i], which means the CPU share is capped at
        the fraction alpha of the total CPU capacity for i.
        """
        return alpha * m.C[i]

    # ===================================================================
    # Parameters
    # ===================================================================
    # Local start time within the chunk for each PDU
    model.τ = Param(
        model.A,
        initialize={j: local_start_time(j, pdu_chunk_dict, chunk_start) for j in model.A}
    )

    # Local activity duration within the chunk for each PDU
    model.ϵ = Param(
        model.A,
        initialize={j: local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end) for j in model.A}
    )

    # Minimum latency for each PDU
    model.l_low = Param(model.A, initialize=pdu_chunk_dict['latency_min'])

    # Maximum latency for each PDU
    model.l_high = Param(model.A, initialize=pdu_chunk_dict['latency_max'])

    # Required rate (Mbps or similar unit) for each PDU
    model.r = Param(model.A, initialize=pdu_chunk_dict['rate'])

    # CPU capacity of each UPF
    model.C = Param(model.U, initialize=upf_dict['cpu_capacity'])

    # Workload factor of each UPF
    model.w = Param(model.U, initialize=upf_dict['workload_factor'])

    # Idle energy consumption for each UPF
    model.E_idle = Param(model.U, initialize=upf_dict['E_idle'])

    # Maximum latency (t_max) for each PDU (same as l_high)
    model.t_max = Param(model.A, initialize=tmax_init)

    # Maximum CPU share for each UPF
    model.s_max = Param(model.U, initialize=smax_init)

    # ===================================================================
    # Decision Variables
    # ===================================================================
    # z[j] indicates if PDU j is admitted (1) or rejected (0)
    model.z = Var(model.A, within=Binary)

    # x[i, n] indicates if UPF i is active at time n
    model.x = Var(model.U, model.T, within=Binary)

    # y[j, i] indicates if PDU j is anchored on UPF i
    model.y = Var(model.A, model.U, within=Binary)

    # s[j, i, n] is the CPU share allocated to PDU j on UPF i at time n
    model.s = Var(model.A, model.U, model.T, within=NonNegativeReals)

    # t[j] is the observed latency for PDU j
    model.t = Var(model.A, within=NonNegativeReals)

    # d[j] is the deviation from the minimum latency for PDU j
    model.d = Var(model.A, within=NonNegativeReals)

    # h[i, n] is the total number of PDU sessions anchored on UPF i at time n
    model.h = Var(model.U, model.T, within=NonNegativeIntegers)

    # phi[j, i, n] is an auxiliary variable linking CPU share, latency, and workload
    model.phi = Var(model.A, model.U, model.T, within=NonNegativeReals)

    # ===================================================================
    # Objective
    # ===================================================================
    def objective_rule(m):
        """
        The objective is to maximize a function that:
          - Rewards the number of admitted sessions (sum(z[j]))
          - Penalizes the number of active UPFs (gamma * sum(x[i,n]))
          - Penalizes the latency deviation (delta * sum(d[j]))
          - Penalizes system energy usage (kappa * (idle + dynamic power across all UPFs))

        Each term is carefully weighted by constants gamma, delta, kappa, etc.
        """
        return (
            # Reward for admitting PDU sessions
            sum(m.z[j] for j in m.A)

            # Penalize total number of active UPFs
            - gamma * sum(m.x[i, n] for i in m.U for n in m.T)

            # Penalize latency deviation from the minimum latency
            - delta * sum(m.d[j] for j in m.A)

            # Penalize total energy usage (both idle and dynamic)
            - kappa * sum(
                m.E_idle[i] * m.x[i, n]  # idle energy cost
                + beta * sum(m.s[j, i, n] for j in m.A) / (alpha * m.C[i])  # dynamic part
                for i in m.U
                for n in m.T
            )
        )
    model.obj = Objective(rule=objective_rule, sense=maximize)

    # ===================================================================
    # Constraints
    # ===================================================================

    # 1) If a PDU j is admitted, it must be anchored on at least one UPF.
    def admittance_constraint(m, j):
        """
        If z[j] == 1 (admitted), then sum of y[j, i] must be >= 1,
        meaning PDU j is anchored on at least one UPF.
        """
        return m.z[j] <= sum(m.y[j, i] for i in m.U)
    model.admittance_constraint = Constraint(model.A, rule=admittance_constraint)

    # 2) Reverse direction to enforce that if z[j] == 0, it cannot be anchored on any UPF.
    def reverse_admittance_constraint(m, j):
        """
        If z[j] == 0 (rejected), then sum of y[j, i] must be 0,
        disallowing anchoring on any UPF.
        """
        return m.z[j] >= sum(m.y[j, i] for i in m.U)
    model.reverse_admittance_constraint = Constraint(model.A, rule=reverse_admittance_constraint)

    # 3) Each PDU can be anchored on at most one UPF.
    def single_assignment_constraint(m, j):
        """
        A PDU j can only be anchored on a single UPF, so the sum of y[j, i] across UPFs i
        must be <= 1.
        """
        return sum(m.y[j, i] for i in m.U) <= 1
    model.single_assignment_constraint = Constraint(model.A, rule=single_assignment_constraint)

    # 4) If a PDU is anchored on UPF i during a time n within its active duration,
    #    then x[i, n] must be 1 (UPF i is active).
    model.activation_constraint = ConstraintList()
    for j in model.A:
        j_start = local_start_time(j, pdu_chunk_dict, chunk_start)
        j_end = j_start + local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end)
        for i in model.U:
            for n in model.T:
                # If time n is within PDU j's active window in this chunk
                if (n >= j_start) and (n <= j_end):
                    # Enforce y[j, i] <= Big-M * x[i, n], effectively linking anchoring to activity
                    model.activation_constraint.add(
                        model.y[j, i] <= M * model.x[i, n]
                    )

    # 5) Scale-in constraint: If x[i, n] = 1, at least one session is using i at n.
    def scale_in_constraint(m, i, n):
        """
        If UPF i is active at time n (x[i,n] == 1), there should be at least one PDU
        anchored to it that is active at time n. This ensures we don't keep UPFs active
        without sessions.
        """
        # Identify which PDUs are relevant at time n
        relevant_sessions = [
            j for j in m.A
            if (local_start_time(j, pdu_chunk_dict, chunk_start) <= n <=
                local_start_time(j, pdu_chunk_dict, chunk_start) +
                local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end))
        ]
        return m.x[i, n] <= sum(m.y[j, i] for j in relevant_sessions)
    model.scale_in_constraint = Constraint(model.U, model.T, rule=scale_in_constraint)

    # 6) CPU capacity constraint: The total CPU share assigned at time n on UPF i
    #    must not exceed alpha*C[i] if UPF i is active, else 0 if inactive.
    def cpu_capacity_constraint(m, i, n):
        """
        Ensures the sum of CPU shares allocated to all relevant PDUs on UPF i at time n
        does not exceed alpha*C[i] * x[i, n]. If x[i, n] = 0, then the sum of CPU shares
        must be 0 for that time on UPF i.
        """
        relevant_sessions = [
            j for j in m.A
            if (local_start_time(j, pdu_chunk_dict, chunk_start) <= n <=
                local_start_time(j, pdu_chunk_dict, chunk_start) +
                local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end))
        ]
        return sum(m.s[j, i, n] for j in relevant_sessions) <= alpha * m.C[i] * m.x[i, n]
    model.cpu_capacity_constraint = Constraint(model.U, model.T, rule=cpu_capacity_constraint)

    # 7) CPU share constraints: The CPU share used by PDU j on UPF i at time n
    #    cannot exceed alpha*C[i] * y[j, i].
    #    We replicate for each time step in the chunk to ensure no overshoot.
    model.cpu_share_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            # Enforce an upper bound for s[j,i,n] across each time n
            for n in model.T:
                model.cpu_share_constraint.add(
                    model.s[j, i, n] <= alpha * model.C[i] * model.y[j, i]
                )

    # 8) Latency bounds constraints: Enforce min <= t[j] <= max for each PDU.
    model.latency_constraint = ConstraintList()
    for j in model.A:
        # Lower bound: t[j] >= l_low[j]
        model.latency_constraint.add(model.l_low[j] <= model.t[j])
        # Upper bound: t[j] <= l_high[j]
        model.latency_constraint.add(model.t[j] <= model.l_high[j])

    # 9) Define bounds on phi[j, i, n] that link w[i]*r[j], t[j], and s[j, i, n].
    model.phi_bounds = ConstraintList()
    for j in model.A:
        j_start = local_start_time(j, pdu_chunk_dict, chunk_start)
        j_end   = j_start + local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end)

        for i in model.U:
            for n in model.T:
                if j_start <= n <= j_end:
                    # Ensure phi[j,i,n] >= w[i]*r[j] if PDU j is anchored on i
                    model.phi_bounds.add(
                        model.w[i]*model.r[j]*model.y[j,i] <= model.phi[j,i,n]
                    )
                    # phi >= t[j] * s[j,i,n] - M*(1-y[j,i])
                    model.phi_bounds.add(
                        model.phi[j,i,n] >= model.t[j]*model.s[j,i,n] - M*(1 - model.y[j,i])
                    )
                    # phi <= t[j]*s_max[i]
                    model.phi_bounds.add(
                        model.phi[j,i,n] <= model.t[j]*model.s_max[i]
                    )
                    # phi <= s[j,i,n]*t_max[j]
                    model.phi_bounds.add(
                        model.phi[j,i,n] <= model.s[j,i,n]*model.t_max[j]
                    )
                else:
                    # If PDU j is not active at time n, phi must be 0 or less
                    model.phi_bounds.add(model.phi[j,i,n] <= 0.0)

    # 10) Deviation constraint: d[j] = t[j] - l_low[j].
    #     This captures how far the latency is from the minimum, used in the objective.
    def deviation_constraint(m, j):
        """
        Defines d[j] as the amount by which t[j] exceeds the minimum latency l_low[j].
        """
        return m.d[j] == m.t[j] - m.l_low[j]
    model.deviation_constraint = Constraint(model.A, rule=deviation_constraint)

    # 11) Session linking constraint: h[i, n] is the count of PDUs anchored on i at time n.
    def session_linking_constraint(m, i, n):
        """
        h[i,n] counts the total number of PDUs j that are anchored on UPF i
        and active at time n.
        """
        relevant_sessions = [
            j for j in m.A
            if (local_start_time(j, pdu_chunk_dict, chunk_start) <= n <=
                local_start_time(j, pdu_chunk_dict, chunk_start) +
                local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end))
        ]
        return m.h[i, n] == sum(m.y[j, i] for j in relevant_sessions)
    model.session_linking_constraint = Constraint(model.U, model.T, rule=session_linking_constraint)

    # 12) No-migration constraint: If a PDU j was anchored on UPF i in a previous chunk,
    #     it must remain on that same UPF i in the current chunk.
    model.nomigration_constraint = ConstraintList()
    for j in model.A:
        if j in previous_assignments:
            assigned_upf = previous_assignments[j]
            # Force y[j, assigned_upf] = 1, and y[j, i != assigned_upf] = 0
            for i in model.U:
                if i == assigned_upf:
                    model.nomigration_constraint.add(model.y[j, i] == 1)
                else:
                    model.nomigration_constraint.add(model.y[j, i] == 0)
            # If it was previously assigned, it was definitely admitted
            model.nomigration_constraint.add(model.z[j] == 1)

    # ===================================================================
    # Solve the model
    # ===================================================================
    results = solver.solve(model, tee=False)

    # Print solver status for diagnostic purposes
    print(f"Chunk [{chunk_start}, {chunk_end}] Solver Status:", results.solver.status)
    print(f"Chunk [{chunk_start}, {chunk_end}] Termination Condition:", results.solver.termination_condition)
    print(f"Chunk [{chunk_start}, {chunk_end}] Objective Value: {model.obj()}")

    # Mark PDUs that were admitted in this chunk so that we do not re-admit them again
    for j in model.A:
        if model.z[j].value > 0.5:
            previously_optimized_pdus.add(j)

    # ===================================================================
    # Record solution for each time step in the chunk
    # ===================================================================
    for n in model.T:
        total_power_at_time_n = 0.0
        for i in model.U:
            # Idle power consumption if UPF i is active at time n
            idle_energy = model.E_idle[i] * model.x[i, n].value
            if model.x[i, n].value > 0.5:
                # Sum of CPU shares for all PDUs anchored on i at time n
                sum_cpu = sum(
                    model.s[j, i, n].value
                    for j in model.A
                    if (local_start_time(j, pdu_chunk_dict, chunk_start) <= n <=
                        local_start_time(j, pdu_chunk_dict, chunk_start) +
                        local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end))
                )
                # Dynamic energy based on CPU usage ratio
                dynamic_energy = beta * sum_cpu / (alpha * model.C[i])
                total_energy = idle_energy + dynamic_energy
            else:
                total_energy = 0.0

            total_power_at_time_n += total_energy

            # Populate solution records for each PDU at each time
            for j in model.A:
                if (local_start_time(j, pdu_chunk_dict, chunk_start) <= n <=
                        local_start_time(j, pdu_chunk_dict, chunk_start) +
                        local_activity_duration(j, pdu_chunk_dict, chunk_start, chunk_end)):
                    solution_records.append({
                        'ChunkStart': chunk_start,
                        'Time': n,
                        'UPF_instance': i,
                        'PDU_session': j,
                        'Admission_status': model.z[j].value,
                        'UPF_active': model.x[i, n].value,
                        'Anchoring': model.y[j, i].value,
                        'CPU_share': model.s[j, i, n].value,
                        'Observed_latency': model.t[j].value,
                        'Idle_energy': idle_energy,
                        'Total_energy': total_energy
                    })

        # Record system-wide power usage for each time
        system_power_records.append({
            'ChunkStart': chunk_start,
            'Time': n,
            'Total_system_power': total_power_at_time_n
        })

    # ===================================================================
    # Update assignments for partially completed PDUs
    # ===================================================================
    # If a PDU extends beyond the current chunk_end, record which UPF it is anchored to
    # so that we enforce no migration in the next chunk.
    new_previous_assignments = {}
    for j in model.A:
        real_end = pdu_chunk_dict['end'][j]
        if real_end > chunk_end and model.z[j].value > 0.5:
            for i in model.U:
                if model.y[j, i].value > 0.5:
                    new_previous_assignments[j] = i
                    break

    # Overwrite previous assignments with updated ones
    previous_assignments = new_previous_assignments

# Convert solution records into a DataFrame and save
results_df = pd.DataFrame(solution_records)
results_df.to_csv('horizon_solution_linear_3600_100_0.5.csv', index=False)

# Convert system power records into a DataFrame and save
system_power_df = pd.DataFrame(system_power_records)
system_power_df.to_csv('system_power_horizon_linear_3600_100_0.5.csv', index=False)

# Final statistics
print("=== Chunked Simulation Complete ===")
total_rejected_sessions = len(results_df[(results_df['Admission_status']==0)])
print(f"Total rejected sessions across all chunks: {total_rejected_sessions}")
