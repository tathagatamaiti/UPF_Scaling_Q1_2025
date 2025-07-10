from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, Objective, ConstraintList, SolverFactory, maximize, NonNegativeReals
)


class OptimizationScaler:
    """
    Optimizer-based scaler for distributing PDUs across UPFs in time chunks.

    At regular intervals (chunks), solves a binary-linear program to:
      - Maximize the number of satisfied PDUs relative to total in chunk,
      - Minimize the number of active UPFs (to reduce cost) weighted by alpha.

    Attributes:
        upfs_df (pd.DataFrame): UPF definitions with cpu_capacity and workload_factor.
        pdus_df (pd.DataFrame): PDU definitions with start_time, end_time, rate, max_latency.
        chunk_interval (int): Time window length for each optimization run.
        alpha (float): Trade-off coefficient for penalizing UPF activation.
        last_check_time (float): Timestamp of last optimization run.
        solver: Pyomo solver instance (SCIP).
        optimized_pdus (set): PDUs already assigned in previous chunks.
        previous_assignments (dict): Mapping of pdu_id to assigned upf_id.
        active_upfs_df (pd.DataFrame): Currently active UPFs after latest solve.
        last_chunk_results (list): Records of assignments from last run.
    """

    def __init__(self, upfs_df, pdus_df, chunk_interval=5, alpha=1.0):
        """
        Initialize the optimizer scaler.

        Args:
            upfs_df (pd.DataFrame): DataFrame of UPFs (upf_id, cpu_capacity, workload_factor).
            pdus_df (pd.DataFrame): DataFrame of PDUs (pdu_id, start_time, end_time, rate, max_latency).
            chunk_interval (int): Duration of each time chunk to optimize over.
            alpha (float): Weight for penalizing active UPFs in objective.
        """
        self.upfs_df = upfs_df
        self.pdus_df = pdus_df
        self.chunk_interval = chunk_interval
        self.alpha = alpha
        # Initialize to ensure first check always runs
        self.last_check_time = -chunk_interval
        # Configure solver (SCIP executable path)
        self.solver = SolverFactory(
            'scip', executable='/home/tmaiti/ups_scaling/SCIPOptSuite-9.2.1-Linux/bin/scip'
        )
        # Track which PDUs have been optimized already
        self.optimized_pdus = set()
        # Store previous assignments to account for overlap across chunks
        self.previous_assignments = {}
        # Active UPFs DataFrame updated each chunk
        self.active_upfs_df = upfs_df.copy()

    def should_check(self, current_time):
        """
        Determine if enough time has passed to trigger another optimization.

        Args:
            current_time (float): Current simulation time.

        Returns:
            bool: True if current_time - last_check_time >= chunk_interval.
        """
        return (current_time - self.last_check_time) >= self.chunk_interval

    def update(self, current_time, active_pdus_dict, pdu_to_upf):
        """
        Run optimization for PDUs active in the upcoming time chunk.

        Builds and solves a Pyomo model that:
          - Decides which UPFs to activate (x[i]).
          - Assigns each PDU j to at most one UPF (y[j,i]).
          - Marks PDUs allocated (z[j]) if assigned.
          - Marks unsatisfied PDUs (v[j]) if their CPU share cannot be fully met.
          - Allocates CPU per PDU-UPF pair (cpu_alloc[j,i]).
        Then extracts assignments and updates active UPF set.

        Args:
            current_time (float): Current simulation time.
            active_pdus_dict (dict): Currently active PDUs (unused here).
            pdu_to_upf (dict): Previous mapping of PDU to UPF (for capacity reservation).
        """
        # Define chunk window
        chunk_start = int(current_time)
        chunk_end = chunk_start + self.chunk_interval
        # Update last check timestamp
        self.last_check_time = current_time

        # Select PDUs that overlap with this chunk and not yet optimized
        active_mask = (
            (self.pdus_df["start_time"] < chunk_end) &
            (self.pdus_df["end_time"] >= chunk_start) &
            (~self.pdus_df["pdu_id"].isin(self.optimized_pdus))
        )
        pdu_chunk_df = self.pdus_df[active_mask].copy()

        # If no new PDUs, simply keep previously active UPFs
        if pdu_chunk_df.empty:
            used_upfs = set(pdu_to_upf.values())
            self.active_upfs_df = self.upfs_df[
                self.upfs_df["upf_id"].isin(used_upfs)
            ].copy()
            return

        # Initialize Pyomo model
        model = ConcreteModel()
        # Sets: A = PDUs in chunk, U = all UPFs
        model.A = Set(initialize=pdu_chunk_df["pdu_id"].tolist())
        model.U = Set(initialize=self.upfs_df["upf_id"].tolist())

        # Decision vars
        model.x = Var(model.U, within=Binary)                   # Activate UPF?
        model.y = Var(model.A, model.U, within=Binary)           # Assign PDU to UPF?
        model.z = Var(model.A, within=Binary)                    # Is PDU allocated?
        model.v = Var(model.A, within=Binary)                    # Is PDU unsatisfied?
        model.cpu_alloc = Var(model.A, model.U, within=NonNegativeReals)  # CPU units allocated

        # Convert DataFrames to dicts for quick lookup
        upf_dict = self.upfs_df.set_index("upf_id").T.to_dict()
        pdu_dict = pdu_chunk_df.set_index("pdu_id").T.to_dict()

        total_pdus = max(1, len(model.A))  # avoid division by zero

        # Objective: maximize allocation minus alpha * active UPFs
        model.obj = Objective(
            expr=(sum(model.z[j] for j in model.A) / total_pdus)
                 - self.alpha * sum(model.x[i] for i in model.U),
            sense=maximize
        )

        model.constraints = ConstraintList()

        # Each PDU: z[j] == sum of y[j,i] <= 1
        for j in model.A:
            model.constraints.add(model.z[j] <= sum(model.y[j, i] for i in model.U))
            model.constraints.add(model.z[j] >= sum(model.y[j, i] for i in model.U))
            model.constraints.add(sum(model.y[j, i] for i in model.U) <= 1)

        # UPF capacity: new + reserved from overlapping previous PDUs
        for i in model.U:
            # Reserve CPU for PDUs still active from previous chunks
            past_cpu = 0.0
            for pdu_id, upf_id in self.previous_assignments.items():
                if upf_id == i:
                    pdu = self.pdus_df.loc[
                        self.pdus_df["pdu_id"] == pdu_id
                    ].iloc[0]
                    if pdu["start_time"] <= chunk_end and pdu["end_time"] >= chunk_start:
                        # Compute CPU requirement for this overlapping PDU
                        past_cpu += (
                            upf_dict[i]["workload_factor"] * pdu["rate"]
                        ) / pdu["max_latency"]
            # Capacity constraint for x[i]
            model.constraints.add(
                sum(model.cpu_alloc[j, i] for j in model.A)
                + past_cpu
                <= upf_dict[i]["cpu_capacity"] * model.x[i]
            )

        # Link cpu_alloc to y and v: allocate <= required if assigned and >= required*(1-v)
        for j in model.A:
            for i in model.U:
                cpu_req = (
                    upf_dict[i]["workload_factor"] * pdu_dict[j]["rate"]
                ) / pdu_dict[j]["max_latency"]
                # If assigned (y=1), cpu_alloc <= cpu_req
                model.constraints.add(model.cpu_alloc[j, i] <= cpu_req * model.y[j, i])
                # If v[j]=0 (satisfied), cpu_alloc >= cpu_req; else cpu_alloc >=0
                model.constraints.add(
                    model.cpu_alloc[j, i]
                    >= cpu_req * (1 - model.v[j]) * model.y[j, i]
                )

        # Solve and suppress solver output
        self.solver.solve(model, tee=False)

        chunk_results = []
        activated_upfs = []
        # Determine which UPFs were activated
        for i in model.U:
            if model.x[i].value > 0.5:
                activated_upfs.append(i)

        # Extract PDU assignments
        for j in model.A:
            if model.z[j].value > 0.5:
                # find the UPF with y[j,i]==1
                assigned = next(
                    i for i in model.U if model.y[j, i].value > 0.5
                )
                cpu = model.cpu_alloc[j, assigned].value
                unsat = model.v[j].value > 0.5
                # Mark PDU as optimized and record assignment
                self.optimized_pdus.add(j)
                self.previous_assignments[j] = assigned
                chunk_results.append({
                    "pdu_id": j,
                    "upf_id": assigned,
                    "cpu_alloc": cpu,
                    "unsatisfied": unsat
                })

        # Update active UPFs DataFrame
        self.active_upfs_df = self.upfs_df[
            self.upfs_df["upf_id"].isin(activated_upfs)
        ].copy()
        # Store last results for PDUScheduler logging
        self.last_chunk_results = chunk_results

    def get_active_upfs(self, current_time=None, upf_allocations=None):
        """
        Retrieve the DataFrame of UPFs active after the last optimization.

        Args:
            current_time: Unused; signature compatible with PDUScheduler.
            upf_allocations: Unused; signature compatible.

        Returns:
            pd.DataFrame: Active UPFs with original columns.
        """
        return self.active_upfs_df
