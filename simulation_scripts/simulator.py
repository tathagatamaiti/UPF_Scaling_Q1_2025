import pandas as pd
import heapq
from typing import Dict, List

from simulation_scripts.HPAscaler import HPAScaler
from simulation_scripts.OptimizationScaler import OptimizationScaler


class PDUScheduler:
    def __init__(self, pdus_file: str, upfs_file: str, mode: str = "static"):
        self.mode = mode
        self.pdus_df = pd.read_csv(pdus_file)
        self.pdus_df["pdu_id"] = self.pdus_df["pdu_id"].astype(int)
        self.pdus_df = self.pdus_df.drop_duplicates(subset="pdu_id").copy()
        assert self.pdus_df["pdu_id"].is_unique, "PDU IDs must be unique"

        self.upfs_df = pd.read_csv(upfs_file)
        self.upfs_df["upf_id"] = self.upfs_df["upf_id"].astype(int)

        if self.mode == "hpa":
            self.hpa = HPAScaler(self.upfs_df)
            self.active_upfs = self.hpa.get_active_upfs()
        elif self.mode == "optimizer":
            self.optimizer = OptimizationScaler(self.upfs_df, self.pdus_df)
            self.active_upfs = self.upfs_df.copy()
        else:
            self.active_upfs = self.upfs_df.copy()

        self.event_queue = []
        self.active_pdus = {}
        self.upf_allocations = {upf_id: {} for upf_id in self.upfs_df["upf_id"]}
        self.pdu_to_upf = {}
        self.pdu_status = {}
        self.result_log: List[Dict] = []
        self.current_time = 0
        self.terminated_pdus = set()
        self.rejected_pdus = set()
        self._initialize_event_queue()

    def _initialize_event_queue(self):
        for _, row in self.pdus_df.iterrows():
            start_event = (row["start_time"], 0, "start", int(row["pdu_id"]))
            end_event = (row["end_time"], 1, "end", int(row["pdu_id"]))
            heapq.heappush(self.event_queue, start_event)
            heapq.heappush(self.event_queue, end_event)

    def _calculate_cpu_share(self, workload_factor, rate, max_latency):
        return (workload_factor * rate) / max_latency

    def _calculate_observed_latency(self, workload_factor, rate, cpu_allocated):
        return (workload_factor * rate) / cpu_allocated if cpu_allocated > 0 else float('inf')

    def _log_allocation(self, pdu_id, upf_id, cpu_allocated, status, event="START"):
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]

        if upf_id is None:
            observed_latency = float('inf')
            required_cpu = 0.0
            upf_info = {"workload_factor": 0}
        else:
            upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
            observed_latency = self._calculate_observed_latency(
                upf["workload_factor"], pdu_row["rate"], cpu_allocated
            )
            required_cpu = self._calculate_cpu_share(
                upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"]
            )
            upf_info = upf

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

    def _rebalance_upf_minimal(self, upf_id):
        allocations = self.upf_allocations[upf_id]
        upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
        total_cpu = upf["cpu_capacity"]

        active_pdu_ids = [pid for pid in allocations if pid in self.active_pdus]
        new_allocations = {}
        total_required = 0.0

        for pid in active_pdu_ids:
            pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pid].iloc[0]
            required = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"])
            new_allocations[pid] = required
            total_required += required

        scale = 1.0
        if total_required > total_cpu:
            scale = total_cpu / total_required

        for pid in new_allocations:
            new_allocations[pid] *= scale
            allocations[pid] = new_allocations[pid]
            self.pdu_status[pid] = "SATISFIED"
            self._log_allocation(pid, upf_id, allocations[pid], "SATISFIED", event="REBALANCE")

    def _rebalance_all_upfs_minimal(self):
        for upf_id in self.active_upfs["upf_id"]:
            self._rebalance_upf_minimal(upf_id)

    def _rebalance_upf_full(self, upf_id):
        allocations = self.upf_allocations[upf_id]
        upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
        total_cpu = upf["cpu_capacity"]

        active_pdu_ids = [pid for pid in allocations if pid in self.active_pdus]
        if not active_pdu_ids:
            return

        requirements = {}
        total_required = 0.0
        for pid in active_pdu_ids:
            pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pid].iloc[0]
            req = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"])
            requirements[pid] = req
            total_required += req

        extra = total_cpu - total_required
        for pid in requirements:
            allocations[pid] = requirements[pid] + (extra / len(requirements) if extra > 0 else 0)
            self._log_allocation(pid, upf_id, allocations[pid], "SATISFIED", event="REBALANCE")

    def _rebalance_all_upfs_full(self):
        for upf_id in self.active_upfs["upf_id"]:
            self._rebalance_upf_full(upf_id)

    def _try_allocate_pdu(self, pdu_id, pdu_row):
        print(
            f"[DEBUG] Attempting to allocate PDU {pdu_id} with rate={pdu_row['rate']}, max_latency={pdu_row['max_latency']}")
        self._rebalance_all_upfs_minimal()

        best_fit_upf = None
        max_remaining_cpu = float('-inf')

        if pdu_id in self.pdu_to_upf:
            assigned_upf = self.pdu_to_upf[pdu_id]
            used_cpu = sum(self.upf_allocations[assigned_upf].values())
            remaining_cpu = max(0.0, self.upfs_df.loc[self.upfs_df["upf_id"] == assigned_upf, "cpu_capacity"].iloc[
                0] - used_cpu)

            required_cpu = self._calculate_cpu_share(
                self.upfs_df.loc[self.upfs_df["upf_id"] == assigned_upf, "workload_factor"].iloc[0], pdu_row["rate"],
                pdu_row["max_latency"])
            if remaining_cpu >= required_cpu:
                best_fit_upf = assigned_upf

        for _, upf in self.active_upfs.iterrows():
            upf_id = upf["upf_id"]
            used_cpu = sum(self.upf_allocations[upf_id].values())
            remaining_cpu = max(0.0, upf["cpu_capacity"] - used_cpu)

            required_cpu = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"])
            if remaining_cpu >= required_cpu:
                best_fit_upf = upf_id
                break

        if best_fit_upf is None:
            inactive_upfs = self.upfs_df[~self.upfs_df["upf_id"].isin(self.active_upfs["upf_id"])]
            for _, upf in inactive_upfs.iterrows():
                upf_id = upf["upf_id"]
                remaining_cpu = upf["cpu_capacity"]
                required_cpu = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"],
                                                         pdu_row["max_latency"])
                if remaining_cpu >= required_cpu:
                    best_fit_upf = upf_id
                    break

        if best_fit_upf is None:
            self.pdu_status[pdu_id] = "REJECTED"
            self.rejected_pdus.add(pdu_id)
            self._log_allocation(pdu_id, None, 0.0, "REJECTED")
            return False

        required_cpu = self._calculate_cpu_share(
            self.upfs_df.loc[self.upfs_df["upf_id"] == best_fit_upf, "workload_factor"].iloc[0], pdu_row["rate"],
            pdu_row["max_latency"])
        self.upf_allocations[best_fit_upf][pdu_id] = required_cpu
        self.pdu_to_upf[pdu_id] = best_fit_upf
        self.pdu_status[pdu_id] = "SATISFIED"
        self._log_allocation(pdu_id, best_fit_upf, required_cpu, "SATISFIED")

        if best_fit_upf not in self.active_upfs["upf_id"].values:
            new_active_upf = self.upfs_df[self.upfs_df["upf_id"] == best_fit_upf]
            self.active_upfs = pd.concat([self.active_upfs, new_active_upf], ignore_index=True)
            print(f"[OPTIMIZER] Activated new UPF {best_fit_upf}")

        self._rebalance_all_upfs_full()
        return True

    def _handle_start(self, pdu_id):
        print(f"[DEBUG] Starting PDU {pdu_id}")
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        success = self._try_allocate_pdu(pdu_id, pdu_row)

        if success and self.pdu_status.get(pdu_id) != "REJECTED":
            self.active_pdus[pdu_id] = pdu_row

    def _handle_end(self, pdu_id):
        print(f"[DEBUG] Ending PDU {pdu_id}")
        if pdu_id not in self.active_pdus:
            print(f"[WARNING] Terminate called for PDU {pdu_id} but it's not in active_pdus!")
            return

        upf_id = self.pdu_to_upf[pdu_id]
        cpu_allocated = self.upf_allocations[upf_id][pdu_id]

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

        del self.upf_allocations[upf_id][pdu_id]
        del self.active_pdus[pdu_id]
        del self.pdu_to_upf[pdu_id]
        del self.pdu_status[pdu_id]
        self.terminated_pdus.add(pdu_id)

        self._rebalance_upf_full(upf_id)

    def run(self):
        while self.event_queue:
            self.current_time, _, event_type, pdu_id = heapq.heappop(self.event_queue)

            if self.mode == "hpa" and self.hpa.should_check(self.current_time):
                self.hpa.update(self.current_time, self.upf_allocations)
                self.active_upfs = self.hpa.get_active_upfs()


            elif self.mode == "optimizer" and (
                    self.current_time - self.optimizer.last_run_time) >= self.optimizer.chunk_interval:
                print(f"[OPTIMIZER] Running optimizer at time {self.current_time}")
                self.optimizer.update(self.current_time, self.active_pdus, self.pdu_to_upf)
                self.active_upfs = self.optimizer.get_active_upfs(self.current_time)
                self.optimizer.last_run_time = self.current_time

                self._rebalance_all_upfs_minimal()

            if event_type == "start":
                self._handle_start(pdu_id)
            elif event_type == "end":
                if pdu_id not in self.rejected_pdus:
                    self._handle_end(pdu_id)

    def export_results(self, file_path="allocation_results.csv"):
        pd.DataFrame(self.result_log).to_csv(file_path, index=False)
        print(f"[INFO] Results saved to {file_path}")