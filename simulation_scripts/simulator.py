import pandas as pd
import heapq
from typing import Dict, List

class PDUScheduler:
    def __init__(self, pdus_file: str, upfs_file: str):
        self.pdus_df = pd.read_csv(pdus_file)
        self.pdus_df["pdu_id"] = self.pdus_df["pdu_id"].astype(int)
        self.pdus_df = self.pdus_df.drop_duplicates(subset="pdu_id").copy()
        assert self.pdus_df["pdu_id"].is_unique, "PDU IDs must be unique"

        self.upfs_df = pd.read_csv(upfs_file)
        self.upfs_df["upf_id"] = self.upfs_df["upf_id"].astype(int)

        self.event_queue = []
        self.active_pdus = {}
        self.upf_allocations = {upf_id: {} for upf_id in self.upfs_df["upf_id"]}
        self.pdu_to_upf = {}
        self.pdu_status = {}
        self.result_log: List[Dict] = []
        self.current_time = 0
        self.terminated_pdus = set()
        self._initialize_event_queue()

    def _initialize_event_queue(self):
        for _, row in self.pdus_df.iterrows():
            start_event = (row["start_time"], 0, "start", int(row["pdu_id"]))
            end_event = (row["end_time"], 1, "end", int(row["pdu_id"]))
            heapq.heappush(self.event_queue, start_event)
            heapq.heappush(self.event_queue, end_event)

        from collections import Counter
        event_counter = Counter(event for _, _, event, _ in self.event_queue)
        print("[DEBUG] Event counts:", event_counter)

    def _calculate_cpu_share(self, workload_factor, rate, max_latency):
        return (workload_factor * rate) / max_latency

    def _calculate_observed_latency(self, workload_factor, rate, cpu_allocated):
        return (workload_factor * rate) / cpu_allocated if cpu_allocated > 0 else float('inf')

    def _log_allocation(self, pdu_id, upf_id, cpu_allocated, status, event="START"):
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        observed_latency = self._calculate_observed_latency(
            self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]["workload_factor"],
            pdu_row["rate"],
            cpu_allocated
        )
        required_cpu = self._calculate_cpu_share(
            self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]["workload_factor"],
            pdu_row["rate"],
            pdu_row["max_latency"]
        )
        self.result_log.append({
            "time": self.current_time,
            "pdu_id": pdu_id,
            "upf_id": upf_id,
            "cpu_allocated": cpu_allocated,
            "observed_latency": observed_latency,
            "required_cpu": required_cpu,
            "required_max_latency": pdu_row["max_latency"],
            "status": status,
            "event": event
        })

    def _try_allocate_pdu(self, pdu_id, pdu_row):
        print(f"[DEBUG] Attempting to allocate PDU {pdu_id} with rate={pdu_row['rate']}, max_latency={pdu_row['max_latency']}")
        best_fit_upf = None
        max_remaining_cpu = float('-inf')

        for _, upf in self.upfs_df.iterrows():
            upf_id = upf["upf_id"]
            used_cpu = sum(self.upf_allocations[upf_id].values())
            remaining_cpu = upf["cpu_capacity"] - used_cpu

            if remaining_cpu > max_remaining_cpu:
                best_fit_upf = upf_id
                max_remaining_cpu = remaining_cpu

        if best_fit_upf is not None:
            print(f"[DEBUG] Best fit UPF for PDU {pdu_id} is {best_fit_upf} with remaining CPU {max_remaining_cpu:.2f}")
            upf = self.upfs_df[self.upfs_df["upf_id"] == best_fit_upf].iloc[0]
            required_cpu = self._calculate_cpu_share(
                upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"]
            )

            allocated_cpu = max(0.0, min(max_remaining_cpu, required_cpu))
            status = "SATISFIED" if allocated_cpu >= required_cpu else "UNSATISFIED"

            self.upf_allocations[best_fit_upf][pdu_id] = allocated_cpu
            self.pdu_to_upf[pdu_id] = best_fit_upf
            self.pdu_status[pdu_id] = status
            self._log_allocation(pdu_id, best_fit_upf, allocated_cpu, status)
            return True

        print(f"[ERROR] No UPF found for PDU {pdu_id}")
        return False

    def _rebalance_upf(self, upf_id):
        allocations = self.upf_allocations[upf_id]
        upf = self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]
        total_cpu = upf["cpu_capacity"]
        exact_requirements = {}

        for pid in allocations:
            pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pid].iloc[0]
            required = self._calculate_cpu_share(upf["workload_factor"], pdu_row["rate"], pdu_row["max_latency"])
            exact_requirements[pid] = required

        used_cpu = sum(exact_requirements.values())
        extra_cpu = total_cpu - used_cpu

        for pid, req in exact_requirements.items():
            if pid not in self.active_pdus:
                continue
            allocated = req
            allocations[pid] = allocated
            status = "SATISFIED" if allocated >= req else "UNSATISFIED"
            self.pdu_status[pid] = status
            self._log_allocation(pid, upf["upf_id"], allocated, status, event="REBALANCE")

        if extra_cpu > 0 and allocations:
            bonus_cpu = extra_cpu / len(allocations)
            for pid in allocations:
                allocations[pid] += bonus_cpu

    def _handle_start(self, pdu_id):
        print(f"[DEBUG] Starting PDU {pdu_id}")
        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        self.active_pdus[pdu_id] = pdu_row
        success = self._try_allocate_pdu(pdu_id, pdu_row)

        if not success:
            print(f"[ERROR] Failed to allocate PDU {pdu_id}, forcing UNSATISFIED with 0 CPU")
            best_fit_upf = self.upfs_df.iloc[0]["upf_id"]
            self.upf_allocations[best_fit_upf][pdu_id] = 0.0
            self.pdu_to_upf[pdu_id] = best_fit_upf
            self.pdu_status[pdu_id] = "UNSATISFIED"
            self._log_allocation(pdu_id, best_fit_upf, 0.0, "UNSATISFIED")

    def _handle_end(self, pdu_id):
        print(f"[DEBUG] Ending PDU {pdu_id}")
        if pdu_id not in self.active_pdus:
            print(f"[WARNING] Terminate called for PDU {pdu_id} but it's not in active_pdus!")
            return

        upf_id = self.pdu_to_upf[pdu_id]
        cpu_allocated = self.upf_allocations[upf_id][pdu_id]

        pdu_row = self.pdus_df[self.pdus_df["pdu_id"] == pdu_id].iloc[0]
        observed_latency = self._calculate_observed_latency(
            self.upfs_df[self.upfs_df["upf_id"] == upf_id].iloc[0]["workload_factor"],
            pdu_row["rate"],
            cpu_allocated
        )
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

        print(f"[DEBUG] Releasing {cpu_allocated:.2f} CPU from PDU {pdu_id} on UPF {upf_id}")
        del self.upf_allocations[upf_id][pdu_id]
        del self.active_pdus[pdu_id]
        del self.pdu_to_upf[pdu_id]
        del self.pdu_status[pdu_id]
        self.terminated_pdus.add(pdu_id)

        self._rebalance_upf(upf_id)

    def run(self):
        print("[INFO] Running simulator")
        while self.event_queue:
            self.current_time, _, event_type, pdu_id = heapq.heappop(self.event_queue)
            if event_type == "start":
                self._handle_start(pdu_id)
            elif event_type == "end":
                self._handle_end(pdu_id)

        if len(self.active_pdus) > 0:
            print("[DEBUG] Active PDUs still remaining at the end:")
            print(list(self.active_pdus.keys()))

    def export_results(self, file_path="allocation_results.csv"):
        print(f"[INFO] Exporting results to {file_path}")
        pd.DataFrame(self.result_log).to_csv(file_path, index=False)
        print(f"[INFO] Results saved to {file_path}")
