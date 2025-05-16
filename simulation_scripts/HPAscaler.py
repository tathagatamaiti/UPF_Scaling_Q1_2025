import pandas as pd
import heapq
from typing import Dict, List

class HPAScaler:
    def __init__(self, upfs_df: pd.DataFrame, scale_out_threshold=70, scale_in_threshold=30, interval=5):
        self.full_upf_pool = upfs_df.copy()
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.interval = interval
        self.time_last_checked = 0
        start_count = max(1, int(0.2 * len(upfs_df)))
        self.active_upfs = upfs_df.iloc[:start_count].copy().reset_index(drop=True)

    def get_active_upfs(self):
        return self.active_upfs

    def should_check(self, current_time):
        return current_time - self.time_last_checked >= self.interval

    def update(self, current_time, upf_allocations: Dict[int, Dict[int, float]]):
        self.time_last_checked = current_time

        utilizations = []
        for _, upf in self.active_upfs.iterrows():
            upf_id = upf["upf_id"]
            used = sum(upf_allocations.get(upf_id, {}).values())
            cap = upf["cpu_capacity"]
            utilizations.append((used / cap) * 100)

        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0

        if avg_util > self.scale_out_threshold and len(self.active_upfs) < len(self.full_upf_pool):
            remaining_upfs = self.full_upf_pool[~self.full_upf_pool["upf_id"].isin(self.active_upfs["upf_id"])]

            util_gap = avg_util - self.scale_out_threshold
            scale_step = 10
            required_count = int(util_gap / scale_step)
            required_count = min(required_count, len(remaining_upfs))

            if required_count > 0:
                to_add = remaining_upfs.iloc[:required_count]
                self.active_upfs = pd.concat([self.active_upfs, to_add], ignore_index=True)
                print(f"[HPA] Scaled OUT: added {len(to_add)} UPFs: {to_add['upf_id'].tolist()}")


        elif avg_util < self.scale_in_threshold and len(self.active_upfs) > 1:
            removable_upfs = []
            for _, upf in self.active_upfs[::-1].iterrows():
                upf_id = upf["upf_id"]
                if not upf_allocations.get(upf_id):
                    removable_upfs.append(upf)
                    break

            if removable_upfs:
                to_remove = removable_upfs[0]
                self.active_upfs = self.active_upfs[self.active_upfs["upf_id"] != to_remove["upf_id"]].reset_index(
                    drop=True)
                print(f"[HPA] Scaled IN: removed UPF {to_remove['upf_id']}")