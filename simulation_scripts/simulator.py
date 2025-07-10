import pandas as pd
import heapq
from typing import Dict, List

from HPAscaler import HPAScaler
from OptimizationScaler import OptimizationScaler


class PDUScheduler:
    """
    Scheduler for Packet Data Units (PDUs) across User Plane Functions (UPFs).

    Depending on the mode, uses static allocation, HPA-based scaling, or an optimizer
    to allocate CPU resources to PDUs while respecting maximum latency constraints.
    """

    def __init__(self, pdus_file: str, upfs_file: str, mode: str = "static"):
        """
        Initialize the PDUScheduler.

        Args:
            pdus_file (str): Path to CSV containing PDU definitions (pdu_id, start_time, end_time, rate, max_latency).
            upfs_file (str): Path to CSV containing UPF definitions (upf_id, cpu_capacity, workload_factor).
            mode (str): Operation mode: "static", "hpa", or "optimizer".
        """
        self.mode = mode
        # Load PDUs and ensure unique integer IDs
        self.pdus_df = pd.read_csv(pdus_file)
        self.pdus_df["pdu_id"] = self.pdus_df["pdu_id"].astype(int)
        self.pdus_df = self.pdus_df.drop_duplicates(subset="pdu_id").copy()
        assert self.pdus_df["pdu_id"].is_unique, "PDU IDs must be unique"

        # Load UPFs and ensure integer IDs
        self.upfs_df = pd.read_csv(upfs_file)
        self.upfs_df["upf_id"] = self.upfs_df["upf_id"].astype(int)

        # Initialize scaling components based on mode
        if self.mode == "hpa":
            self.hpa = HPAScaler(self.upfs_df)
            self.active_upfs = self.hpa.get_active_upfs()
        elif self.mode == "optimizer":
            self.optimizer = OptimizationScaler(self.upfs_df, self.pdus_df)
            self.active_upfs = self.upfs_df.copy()
        else:
            # static mode: all UPFs are active initially
            self.active_upfs = self.upfs_df.copy()

        # Event queue for simulation: (time, priority, event_type, pdu_id)
        self.event_queue = []
        # Track currently active PDUs
        self.active_pdus = {}
        # CPU allocations per UPF: {upf_id: {pdu_id: cpu_alloc}}
        self.upf_allocations = {upf_id: {} for upf_id in self.upfs_df["upf_id"]}
        # Mapping of PDU to assigned UPF
        self.pdu_to_upf = {}
        # Status of each PDU (SATISFIED, UNSATISFIED, REJECTED)
        self.pdu_status = {}
        # Log of allocation events for analysis
        self.result_log: List[Dict] = []
        # Simulation clock
        self.current_time = 0
        # Sets for terminated and rejected PDUs
        self.terminated_pdus = set()
        self.rejected_pdus = set()
        # Populate the event queue based on PDU start/end times
        self._initialize_event_queue()

    def _initialize_event_queue(self):
        """
        Push start and end events for each PDU into the priority queue.
        """
        for _, row in self.pdus_df.iterrows():
            # start event has higher priority (0) than end event (1)
            start_event = (row["start_time"], 0, "start", int(row["pdu_id"]))
            end_event = (row["end_time"], 1, "end", int(row["pdu_id"]))
            heapq.heappush(self.event_queue, start_event)
            heapq.heappush(self.event_queue, end_event)

    def _calculate_cpu_share(self, workload_factor, rate, max_latency):
        """
        Compute the CPU capacity required to meet latency for a given PDU.

        Args:
            workload_factor (float): Processing factor of the UPF.
            rate (float): Packet rate of the PDU.
            max_latency (float): Maximum tolerated latency for the PDU.

        Returns:
            float: CPU units needed.
        """
        return (workload_factor * rate) / max_latency

    def _calculate_observed_latency(self, workload_factor, rate, cpu_allocated):
        """
        Compute the observed latency given CPU allocation.

        If no CPU is allocated, returns infinite latency.
        """
        return (workload_factor * rate) / cpu_allocated if cpu_allocated > 0 else float('inf')

    def _log_allocation(self, pdu_id, upf_id, cpu_allocated, status, event="START"):
        """
        Record an allocation or status change to the result log.
        """
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]

        if upf_id is None:
            # No UPF assigned => infinite latency
            observed_latency = float('inf')
            required_cpu = 0.0
        else:
            upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
            observed_latency = self._calculate_observed_latency(
                upf["workload_factor"], pdu_row["rate"], cpu_allocated
            )
            required_cpu = self._calculate_cpu_share(
                upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"]
            )

        # Append the record with relevant metrics
        self.result_log.append({
            "time": self.current_time,
            "pdu_id": pdu_id,
            "upf_id": upf_id if upf_id is not None else "None",
            "cpu_allocated": cpu_allocated,
            "observed_latency": observed_latency,
            "required_cpu": required_cpu,
            "required_max_latency": pdu_row["max_latency"],
            "status": status,
            "event": event
        })

    def _try_allocate_pdu(self, pdu_id, pdu_row):
        """
        Attempt to allocate a PDU to the best-fitting UPF based on remaining CPU.

        Returns:
            bool: True if allocation succeeded, False if rejected.
        """
        best_fit_upf = None

        # If in optimizer mode and PDU was already mapped, try same UPF first
        if self.mode == "optimizer" and pdu_id in self.pdu_to_upf:
            assigned_upf = self.pdu_to_upf[pdu_id]
            used_cpu = sum(self.upf_allocations[assigned_upf].values())
            remaining_cpu = max(0.0,
                                self.upfs_df.loc[self.upfs_df["upf_id"] == assigned_upf, "cpu_capacity"].iloc[
                                    0] - used_cpu)
            required_cpu = self._calculate_cpu_share(
                self.upfs_df.loc[self.upfs_df["upf_id"] == assigned_upf, "workload_factor"].iloc[0],
                pdu_row["rate"], pdu_row["max_latency"]
            )
            if remaining_cpu >= required_cpu:
                best_fit_upf = assigned_upf

        # Update remaining CPU stats for active UPFs
        self.active_upfs["remaining_cpu"] = self.active_upfs["upf_id"].apply(
            lambda uid: self.upfs_df.loc[self.upfs_df["upf_id"] == uid, "cpu_capacity"].iloc[0] -
                        sum(self.upf_allocations.get(uid, {}).values())
        )
        # Sort by most available CPU
        active_upfs_sorted = self.active_upfs.sort_values(by="remaining_cpu", ascending=False)

        # Find first UPF that can host this PDU
        for _, upf in active_upfs_sorted.iterrows():
            upf_id = upf["upf_id"]
            used_cpu = sum(self.upf_allocations[upf_id].values())
            remaining_cpu = max(0.0, upf["cpu_capacity"] - used_cpu)
            required_cpu = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"])
            if remaining_cpu >= required_cpu:
                best_fit_upf = upf_id
                break

        # If no active UPF fits, try any inactive UPFs
        if best_fit_upf is None:
            inactive_upfs = self.upfs_df[~self.upfs_df["upf_id"].isin(self.active_upfs["upf_id"])]
            for _, upf in inactive_upfs.iterrows():
                upf_id = upf["upf_id"]
                required_cpu = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"],
                                                         pdu_row["max_latency"])
                if upf["cpu_capacity"] >= required_cpu:
                    best_fit_upf = upf_id
                    break

        # If still none, reject the PDU
        if best_fit_upf is None:
            self.pdu_status[pdu_id] = "REJECTED"
            self.rejected_pdus.add(pdu_id)
            self._log_allocation(pdu_id, None, 0.0, "REJECTED")
            return False

        # Allocate and log success
        required_cpu = self._calculate_cpu_share(
            self.upfs_df.loc[self.upfs_df["upf_id"] == best_fit_upf, "workload_factor"].iloc[0],
            pdu_row["rate"], pdu_row["max_latency"]
        )
        self.upf_allocations[best_fit_upf][pdu_id] = required_cpu
        self.pdu_to_upf[pdu_id] = best_fit_upf
        self.pdu_status[pdu_id] = "SATISFIED"
        self._log_allocation(pdu_id, best_fit_upf, required_cpu, "SATISFIED")

        # Add new UPF to active set if needed
        if best_fit_upf not in self.active_upfs["upf_id"].values:
            new_active_upf = self.upfs_df[self.upfs_df["upf_id"] == best_fit_upf]
            self.active_upfs = pd.concat([self.active_upfs, new_active_upf], ignore_index=True)

        return True

    def _redistribute_upf_cpu(self, upf_id):
        """
        When optimization frees or adds capacity, redistribute CPU among PDUs on a UPF.
        """
        pdu_ids = list(self.upf_allocations[upf_id].keys())
        if not pdu_ids:
            return

        upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
        total_capacity = upf["cpu_capacity"]

        under_provisioned = []
        total_required_deficit = 0.0
        pdu_rows = self.pdus_df[self.pdus_df["pdu_id"].isin(pdu_ids)]

        # Identify PDUs needing extra CPU
        for _, pdu_row in pdu_rows.iterrows():
            pdu_id = pdu_row["pdu_id"]
            required_cpu = self._calculate_cpu_share(
                upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"]
            )
            current_cpu = self.upf_allocations[upf_id][pdu_id]

            if current_cpu < required_cpu:
                deficit = required_cpu - current_cpu
                under_provisioned.append((pdu_id, current_cpu, required_cpu, deficit))
                total_required_deficit += deficit

        if total_required_deficit == 0:
            return

        available_cpu = total_capacity - sum(self.upf_allocations[upf_id].values())
        if available_cpu <= 0:
            return

        # Distribute available CPU proportionally to deficits
        for pdu_id, current_cpu, required_cpu, deficit in under_provisioned:
            share = (deficit / total_required_deficit) * available_cpu
            new_cpu = current_cpu + share
            self.upf_allocations[upf_id][pdu_id] = new_cpu

            pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
            observed_latency = self._calculate_observed_latency(
                upf["workload_factor"], pdu_row["rate"], new_cpu
            )
            status = (
                "SATISFIED" if observed_latency <= pdu_row["max_latency"] else "UNSATISFIED"
            )
            self.pdu_status[pdu_id] = status
            self._log_allocation(pdu_id, upf_id, new_cpu, status, event="REDISTRIBUTE")

    def _handle_start(self, pdu_id):
        """
        Handle a PDU start event: attempt allocation and track active PDUs.
        """
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        success = self._try_allocate_pdu(pdu_id, pdu_row)
        if success and self.pdu_status.get(pdu_id) != "REJECTED":
            self.active_pdus[pdu_id] = pdu_row

    def _handle_end(self, pdu_id):
        """
        Handle a PDU end event: free resources and record termination.
        """
        if pdu_id not in self.active_pdus:
            return  # skip if never active or was rejected

        upf_id = self.pdu_to_upf[pdu_id]
        cpu_allocated = self.upf_allocations[upf_id][pdu_id]

        # Log termination event
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
        observed_latency = self._calculate_observed_latency(upf["workload_factor"], pdu_row["rate"], cpu_allocated)

        self.result_log.append({
            "time": self.current_time,
            "pdu_id": pdu_id,
            "upf_id": upf_id,
            "cpu_allocated": cpu_allocated,
            "observed_latency": observed_latency,
            "required_max_latency": pdu_row["max_latency"],
            "status": self.pdu_status[pdu_id],
            "event": "TERMINATE"
        })

        # Release resources
        del self.upf_allocations[upf_id][pdu_id]
        del self.active_pdus[pdu_id]
        del self.pdu_to_upf[pdu_id]
        del self.pdu_status[pdu_id]
        self.terminated_pdus.add(pdu_id)

        # Rebalance if in optimizer mode
        if self.mode == "optimizer":
            self._redistribute_upf_cpu(upf_id)

    def run(self):
        """
        Run the scheduling simulation by processing chronological events.
        """
        while self.event_queue:
            # Pop next event
            self.current_time, _, event_type, pdu_id = heapq.heappop(self.event_queue)

            # Check scaling triggers for HPA or optimizer modes
            if self.mode == "hpa" and self.hpa.should_check(self.current_time):
                self.hpa.update(self.current_time, self.upf_allocations)
                self.active_upfs = self.hpa.get_active_upfs()

            elif self.mode == "optimizer" and self.optimizer.should_check(self.current_time):
                self.optimizer.update(self.current_time, self.active_pdus, self.pdu_to_upf)
                self.active_upfs = self.optimizer.get_active_upfs(self.current_time, self.upf_allocations)
                # Log any elastic reallocation actions
                for rec in getattr(self.optimizer, "last_chunk_results", []):
                    pdu = rec["pdu_id"]
                    upf = rec["upf_id"]
                    cpu = rec["cpu_alloc"]
                    status = "UNSATISFIED" if rec["unsatisfied"] else "SATISFIED"
                    self._log_allocation(pdu, upf, cpu, status, event="ELASTIC")

            # Dispatch event handling
            if event_type == "start":
                self._handle_start(pdu_id)
            elif event_type == "end":
                if pdu_id not in self.rejected_pdus:
                    self._handle_end(pdu_id)

    def get_results_df(self) -> pd.DataFrame:
        """
        Retrieve a DataFrame containing all logged allocation events.

        Returns:
            pd.DataFrame: Columns include time, pdu_id, upf_id, cpu_allocated, observed_latency,
            required_max_latency, status, event, etc.
        """
        return pd.DataFrame(self.result_log)


def compute_active_upf_utilization(results_df: pd.DataFrame, upfs_df: pd.DataFrame) -> float:
    """
    Compute the average utilization percentage across active UPFs over time.

    Args:
        results_df (pd.DataFrame): Log of allocation and termination events.
        upfs_df (pd.DataFrame): Static UPF definitions with capacities.

    Returns:
        float: Average utilization as percent.
    """
    upf_capacities = dict(zip(upfs_df["upf_id"], upfs_df["cpu_capacity"]))

    # Extract start and terminate times
    start_events = results_df[results_df["event"] == "START"]
    end_events = results_df[results_df["event"] == "TERMINATE"]

    # Merge to get session durations per PDU
    session_df = pd.merge(
        start_events[["pdu_id", "time", "upf_id", "cpu_allocated"]],
        end_events[["pdu_id", "time"]],
        on="pdu_id",
        suffixes=("_start", "_end")
    )

    time_points = sorted(results_df["time"].unique())
    utilization_over_time = []

    for t in time_points:
        # Find PDUs active at time t
        snapshot = session_df[(session_df["time_start"] <= t) & (session_df["time_end"] > t)]

        if snapshot.empty:
            utilization_over_time.append(0)
            continue

        utilization_sum = 0
        active_upf_count = 0

        for upf_id in snapshot["upf_id"].unique():
            upf_snapshot = snapshot[snapshot["upf_id"] == upf_id]
            used_cpu = upf_snapshot["cpu_allocated"].sum()
            capacity = upf_capacities.get(upf_id, 1e-9)

            utilization = min(used_cpu / capacity, 1.0)
            utilization_sum += utilization
            active_upf_count += 1

        # Average across active UPFs at this time point
        utilization_percent = (utilization_sum / active_upf_count) * 100 if active_upf_count else 0
        utilization_over_time.append(utilization_percent)

    # Compute overall average
    return sum(utilization_over_time) / len(utilization_over_time) if utilization_over_time else 0.0


def generate_summary_from_results(results_df: pd.DataFrame, normalized_latency: float) -> pd.DataFrame:
    """
    Build a summary DataFrame reporting PDU fulfillment statistics.
    """
    df_start = results_df[results_df["event"] == "START"]
    total_pdus = len(df_start)
    satisfied = (df_start["status"] == "SATISFIED").sum()
    unsatisfied = (df_start["status"] == "UNSATISFIED").sum()
    rejected = (df_start["status"] == "REJECTED").sum()
    used_upfs = df_start[df_start["status"] != "REJECTED"]["upf_id"].nunique()

    return pd.DataFrame([{
        "total_pdus": total_pdus,
        "satisfied_pdus": satisfied,
        "unsatisfied_pdus": unsatisfied,
        "rejected_pdus": rejected,
        "used_upfs": used_upfs,
        "satisfaction_ratio (%)": round((satisfied / total_pdus) * 100, 2) if total_pdus > 0 else 0,
        "avg_normalized_latency": round(normalized_latency, 4)
    }])


def compute_average_normalized_latency(results_df: pd.DataFrame) -> float:
    """
    Calculate the mean of observed latency normalized by the required max latency for all satisfied or unsatisfied PDUs.
    """
    df_start = results_df[results_df["event"] == "START"].copy()
    df_start = df_start[df_start["status"].isin(["SATISFIED", "UNSATISFIED"])]
    df_start = df_start[df_start["required_max_latency"] > 0]

    if df_start.empty:
        return 0.0

    df_start["normalized_latency"] = df_start["observed_latency"] / df_start["required_max_latency"]
    return df_start["normalized_latency"].mean()


def main():
    """
    Entry point: run simulation for each mode and output utilization and summary stats.
    """

    # Load input data
    pdus_df = pd.read_csv("data/input/pdus.csv")
    upfs_df = pd.read_csv("data/input/upfs.csv")

    modes = ["static", "hpa", "optimizer"]
    for mode in modes:
        scheduler = PDUScheduler(pdus_df, upfs_df, mode=mode)
        scheduler.run()
        results_df = scheduler.get_results_df()

        utilization = compute_active_upf_utilization(results_df, upfs_df)
        normalized_latency = compute_average_normalized_latency(results_df)
        summary_df = generate_summary_from_results(results_df, normalized_latency)

        # Print results to console
        print(f"\n[{mode.upper()}] Avg Active UPF Utilization: {utilization:.2f}%")
        print(summary_df)


if __name__ == "__main__":
    main()
