from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, Objective, ConstraintList, SolverFactory, maximize, NonNegativeReals
)

class OptimizationScaler:
    def __init__(self, upfs_df, pdus_df, chunk_interval=15):
        self.upfs_df = upfs_df
        self.pdus_df = pdus_df
        self.chunk_interval = chunk_interval
        self.last_check_time = -chunk_interval
        self.solver = SolverFactory('scip', executable='/home/tmaiti/ups_scaling/SCIPOptSuite-9.2.1-Linux/bin/scip')
        self.optimized_pdus = set()
        self.previous_assignments = {}
        self.active_upfs_df = upfs_df.copy()
        self.last_run_time = -float('inf')

    def should_check(self, current_time):
        return current_time - self.last_check_time >= self.chunk_interval

    def get_active_upfs(self, current_time):
        active_upfs_sorted = self.active_upfs_df.copy()
        active_upfs_sorted["used_cpu"] = [
            sum(self.upf_allocations.get(upf_id, {}).values())
            for upf_id in active_upfs_sorted["upf_id"]
        ]
        active_upfs_sorted = active_upfs_sorted.sort_values(by="used_cpu", ascending=False)
        return active_upfs_sorted

    def update(self, current_time, active_pdus_dict, pdu_to_upf):
        chunk_start = int(current_time)
        chunk_end = chunk_start + self.chunk_interval - 1
        self.last_check_time = current_time

        active_mask = (
            (self.pdus_df["start_time"] <= chunk_end)
            & (self.pdus_df["end_time"] >= chunk_start)
            & (~self.pdus_df["pdu_id"].isin(self.optimized_pdus))
        )
        pdu_chunk_df = self.pdus_df[active_mask].copy()

        if pdu_chunk_df.empty:
            self.active_upfs_df = self.upfs_df[self.upfs_df["upf_id"].isin(set(pdu_to_upf.values()))].copy()
            return

        model = ConcreteModel()
        model.A = Set(initialize=pdu_chunk_df["pdu_id"].tolist())
        model.U = Set(initialize=self.upfs_df["upf_id"].tolist())
        model.x = Var(model.U, within=Binary)
        model.y = Var(model.A, model.U, within=Binary)
        model.z = Var(model.A, within=Binary)
        model.cpu_alloc = Var(model.A, model.U, within=NonNegativeReals)
        model.v = Var(model.A, within=Binary)

        upf_dict = self.upfs_df.set_index("upf_id").to_dict("index")
        pdu_dict = pdu_chunk_df.set_index("pdu_id").to_dict("index")

        total_pdus = len(model.A) if len(model.A) > 0 else 1

        model.obj = Objective(
            expr=(sum(model.z[j] for j in model.A) / total_pdus)
                 - 1 * (sum(model.x[i] for i in model.U))
                 - 1 * (sum(model.v[j] for j in model.A)),
            sense=maximize
        )

        model.constraints = ConstraintList()

        for j in model.A:
            model.constraints.add(model.z[j] == sum(model.y[j, i] for i in model.U))
            model.constraints.add(sum(model.y[j, i] for i in model.U) <= 1)

        for i in model.U:
            model.constraints.add(
                sum(model.cpu_alloc[j, i] for j in model.A) <= upf_dict[i]["cpu_capacity"] * model.x[i]
            )

        for j in model.A:
            for i in model.U:
                cpu_req = (upf_dict[i]["workload_factor"] * pdu_dict[j]["rate"]) / pdu_dict[j]["max_latency"]
                model.constraints.add(model.cpu_alloc[j, i] <= cpu_req * model.y[j, i])
                model.constraints.add(
                    sum(model.cpu_alloc[j, i] for i in model.U) >= cpu_req * (1 - model.v[j])
                )

        self.solver.solve(model, tee=False)

        activated_upfs = []
        for i in model.U:
            if model.x[i].value > 0.5:
                activated_upfs.append(i)

        admitted_pdus = 0
        for j in model.A:
            if model.z[j].value and model.z[j].value > 0.5:
                self.optimized_pdus.add(j)
                admitted_pdus += 1
                for i in model.U:
                    if model.y[j, i].value and model.y[j, i].value > 0.5:
                        self.previous_assignments[j] = i
                        break

        self.active_upfs_df = self.upfs_df[self.upfs_df["upf_id"].isin(activated_upfs)].copy()
        print(f"[OPTIMIZER] Chunk {chunk_start}-{chunk_end}: considered {len(pdu_chunk_df)}, "
              f"admitted {admitted_pdus}")
