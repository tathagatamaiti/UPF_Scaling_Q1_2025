import pandas as pd
import math

# =======================================================================
# Data Loading
# =======================================================================
pdu_data = pd.read_csv('pdu_sessions_3600_42_2.csv')  # PDU session dataset
upf_data = pd.read_csv('upf_instances_100_42.csv')  # UPF instances dataset

# =======================================================================
# Simulation Parameters
# =======================================================================
alpha = 0.8  # Fraction of UPF CPU capacity that can be allocated to PDUs
beta = 1  # Weight factor for dynamic energy consumption
scale_out_threshold = 0.6  # Threshold to trigger scaling out (activating more UPFs)
scale_in_threshold = 0.4  # Threshold to trigger scaling in (deactivating some UPFs)
time_interval = 15  # Interval (in time units) for each simulation step
end_time = 4000  # Maximum simulation time

start_time = 15  # Start time for the simulation
time_points = list(range(start_time, math.ceil(end_time) + 1, time_interval))

# =======================================================================
# Initial Conditions and Data Structures
# =======================================================================
active_upfs = [0]  # List of indices for currently active UPFs (start with UPF #0)
max_upfs = len(upf_data)  # Maximum number of UPFs available

results = []  # Will store simulation results for each time step
system_power = []  # Will store system power usage for each time step
pdu_allocations = {}  # Map of PDU -> currently assigned UPF
global_satisfied_pdus = 0  # Count of globally satisfied PDUs
global_unsatisfied_pdus = 0  # Count of globally unsatisfied PDUs


# =======================================================================
# Helper Functions
# =======================================================================
def min_cpu_required(workload_factor, rate, latency_max):
    """
    Compute the minimum CPU share required for a given PDU to meet its maximum latency constraint.

    This function calculates how much CPU share is needed for a PDU session based on:
      - The UPF's workload factor.
      - The PDU's required data rate (rate).
      - The PDU's maximum allowable latency (latency_max).

    Parameters
    ----------
    workload_factor : float
        A numeric factor indicating how the UPF translates workload (rate) into CPU usage needed
        to maintain a certain latency range.
    rate : float
        The rate requirement (e.g., Mbps or a similar unit) for the PDU session.
    latency_max : float
        The maximum latency the PDU session can tolerate.

    Returns
    -------
    float
        The minimum CPU share required such that the session meets the latency requirement.
    """
    return (workload_factor * rate) / latency_max


# =======================================================================
# Main Simulation Loop
# =======================================================================
for n in time_points:
    """
    For each time point n in the simulation:
      1. Identify which PDUs are active around this time interval.
      2. Attempt to allocate or keep PDUs on active UPFs, respecting CPU constraints.
      3. Record the distribution of CPU shares among PDUs.
      4. Potentially scale out or scale in the UPF pool based on total workload vs. thresholds.
      5. Compute total power usage at this time step and store results.
    """

    # Identify PDUs active within the last 'time_interval' window ending at time n.
    # Condition: (start < n) & (end >= n - time_interval)
    # This means the PDU started before time n and hasn't ended before time n - time_interval.
    # Essentially, it catches PDUs that are active in the current time window.
    active_pdus = pdu_data[
        (pdu_data['start'] < n) & (pdu_data['end'] >= n - time_interval)
        ]['id'].tolist()

    # If there are no active PDUs, move to the next time point.
    if not active_pdus:
        continue

    # upf_allocations: dictionary mapping each active UPF -> list of PDUs assigned to it.
    upf_allocations = {upf: [] for upf in active_upfs}

    # pdu_cpu_shares: dictionary storing how much CPU share is allocated to each active PDU.
    pdu_cpu_shares = {pdu: 0 for pdu in active_pdus}

    # pdu_latency_status: dictionary storing "SATISFIED" or "UNSATISFIED" latency status for each PDU.
    pdu_latency_status = {}

    # -------------------------------------------------------------------
    # Allocation Logic
    # -------------------------------------------------------------------
    for pdu in active_pdus:
        # If this PDU has been assigned before, check if its assigned UPF is still active.
        if pdu in pdu_allocations:
            assigned_upf = pdu_allocations[pdu]
            # If the previously assigned UPF is still active, re-assign this PDU there directly.
            if assigned_upf in active_upfs:
                upf_allocations[assigned_upf].append(pdu)
                continue
            else:
                # If the previously assigned UPF is no longer active, remove that assignment info.
                del pdu_allocations[pdu]

        # Extract relevant PDU and UPF parameters from the dataframe
        rate = pdu_data.loc[pdu_data['id'] == pdu, 'rate'].values[0]
        latency_max = pdu_data.loc[pdu_data['id'] == pdu, 'latency_max'].values[0]
        workload_factor = upf_data['workload_factor'].iloc[0]

        # Calculate minimum required CPU for this PDU to meet its latency target.
        min_cpu = min_cpu_required(workload_factor, rate, latency_max)

        allocated = False  # Will be set to True if we successfully find a UPF for this PDU

        # Try to allocate the PDU to one of the currently active UPFs
        for upf in upf_allocations:
            # Compute how much CPU is already allocated to PDUs on this UPF
            current_sum_cpu = sum(pdu_cpu_shares[p] for p in upf_allocations[upf])
            capacity_upf = upf_data.loc[upf, 'cpu_capacity'] * alpha

            # If adding this PDU's min_cpu stays within the capacity limit, we allocate here
            if current_sum_cpu + min_cpu <= capacity_upf:
                upf_allocations[upf].append(pdu)
                pdu_cpu_shares[pdu] = min_cpu
                pdu_latency_status[pdu] = 'SATISFIED'
                pdu_allocations[pdu] = upf
                global_satisfied_pdus += 1
                allocated = True
                break

        # If we didn't manage to allocate within the UPF capacity constraints,
        # we forcibly place the PDU on the first active UPF, marking it as "UNSATISFIED".
        if not allocated:
            first_upf = active_upfs[0]
            upf_allocations[first_upf].append(pdu)
            pdu_cpu_shares[pdu] = min_cpu
            pdu_latency_status[pdu] = 'UNSATISFIED'
            pdu_allocations[pdu] = first_upf
            global_unsatisfied_pdus += 1

    # -------------------------------------------------------------------
    # CPU Redistribution Logic (Extra CPU to PDUs)
    # -------------------------------------------------------------------
    for upf, pdus in upf_allocations.items():
        total_cpu_capacity = upf_data.loc[upf, 'cpu_capacity'] * alpha
        total_cpu_used = sum(pdu_cpu_shares[pdu] for pdu in pdus)

        # If there's unused capacity (total_cpu_used < total_cpu_capacity)
        # and the UPF has assigned PDUs, distribute the remaining CPU share equally.
        if total_cpu_used < total_cpu_capacity and pdus:
            extra_cpu = (total_cpu_capacity - total_cpu_used) / len(pdus)
            for pdu in pdus:
                pdu_cpu_shares[pdu] += extra_cpu

    # -------------------------------------------------------------------
    # Scaling Logic
    # -------------------------------------------------------------------
    # Compute the total workload across all active UPFs
    total_workload = sum(
        pdu_cpu_shares[pdu]
        for upf in upf_allocations
        for pdu in upf_allocations[upf]
    )
    # Compute the current total CPU capacity of all active UPFs combined
    current_capacity = len(active_upfs) * upf_data['cpu_capacity'].iloc[0] * alpha

    # -- Scale Out Condition --
    # If the total workload exceeds a fraction 'scale_out_threshold' of current capacity,
    # we activate enough new UPFs to handle the workload.
    if total_workload > scale_out_threshold * current_capacity:
        required_upfs = math.ceil(total_workload / (upf_data['cpu_capacity'].iloc[0] * alpha))
        # Ensure we do not exceed the total number of UPFs available
        required_upfs = min(required_upfs, max_upfs)

        # Number of additional UPFs needed
        new_upfs = min(required_upfs - len(active_upfs), max_upfs - len(active_upfs))

        # Extend the active_upfs list with new UPFs by their indices
        active_upfs.extend(range(len(active_upfs), len(active_upfs) + new_upfs))

    # -- Scale In Condition --
    # If total workload is below 'scale_in_threshold' times current capacity,
    # and there is more than one UPF active, scale down.
    elif total_workload < scale_in_threshold * current_capacity and len(active_upfs) > 1:
        # The new size after scaling in is enough to just meet total_workload
        new_size = math.ceil(total_workload / (upf_data['cpu_capacity'].iloc[0] * alpha))
        # Retain only that many UPFs (this might not perfectly track specific IDs in advanced usage)
        active_upfs = active_upfs[:new_size]

    # -------------------------------------------------------------------
    # Power Computation
    # -------------------------------------------------------------------
    # Idle energy for each active UPF: upf_data['E_idle'].iloc[0]
    # multiplied by the number of active UPFs (assuming identical E_idle for all).
    idle_energy = sum(upf_data['E_idle'].iloc[0] for _ in active_upfs)

    # Dynamic energy is proportional to the ratio of used CPU to total CPU capacity
    # times the factor 'beta'.
    dynamic_energy = beta * (total_workload / (upf_data['cpu_capacity'].iloc[0] * alpha))

    # Total power = idle + dynamic energy
    total_power = idle_energy + dynamic_energy

    # -------------------------------------------------------------------
    # Recording Results
    # -------------------------------------------------------------------
    results.append({
        'Time_instance': n,  # The current simulation time
        'Active_PDUs': len(active_pdus),  # Number of PDUs active in this interval
        'Active_UPFs': len(active_upfs),  # Number of UPFs currently active
        'Total_power': total_power,  # Total power consumption (idle + dynamic)
        'Total_CPU_Usage': total_workload,  # Sum of CPU shares allocated to PDUs
        'Active_UPF_Ids': list(active_upfs),  # The list of UPF IDs currently active
        'PDU_Latency_Status': pdu_latency_status  # Dictionary of each PDU's latency satisfaction
    })

    system_power.append({
        'Time_instance': n,
        'Total_system_power': total_power
    })

# =======================================================================
# Final Data Persistence
# =======================================================================
results_df = pd.DataFrame(results)
results_df.to_csv('hpa_3600_100_2_60_40.csv', index=False)

system_power_df = pd.DataFrame(system_power)
system_power_df.to_csv('system_power_hpa_3600_100_2_60_40.csv', index=False)

# =======================================================================
# Summary
# =======================================================================
print(f"Total Time Instances Simulated: {len(results)}")
print(f"Total Satisfied PDUs: {global_satisfied_pdus}")
print(f"Total Unsatisfied PDUs: {global_unsatisfied_pdus}")
print("HPA simulation completed.")
