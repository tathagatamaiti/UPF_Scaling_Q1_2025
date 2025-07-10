import pandas as pd
from typing import Dict


class HPAScaler:
    """
    Horizontal Pod Autoscaler (HPA) style scaler for UPFs based on CPU utilization.

    Periodically checks average and maximum CPU utilization of active UPFs:
      - Scales out (adds UPFs) when max utilization exceeds scale_out_threshold.
      - Scales in (removes idle UPFs) when average utilization falls below scale_in_threshold.

    Attributes:
        full_upf_pool (pd.DataFrame): DataFrame of all available UPFs.
        scale_out_threshold (float): % CPU utilization to trigger scale-out.
        scale_in_threshold (float): % CPU utilization to trigger scale-in.
        interval (int): Time interval (same units as PDU times) between checks.
        time_last_checked (float): Last time an update was performed.
        active_upfs (pd.DataFrame): Currently active UPFs after last update.
    """

    def __init__(
        self,
        upfs_df: pd.DataFrame,
        scale_out_threshold: float = 70,
        scale_in_threshold: float = 30,
        interval: int = 5
    ):
        """
        Initialize the HPA scaler.

        Args:
            upfs_df (pd.DataFrame): DataFrame of UPFs including 'upf_id' and 'cpu_capacity'.
            scale_out_threshold (float): Utilization % above which to add UPFs.
            scale_in_threshold (float): Utilization % below which to remove UPFs.
            interval (int): Time units between utilization checks.
        """
        # Store the full pool of possible UPFs
        self.full_upf_pool = upfs_df.copy()
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.interval = interval
        # Initialize last checked to time zero
        self.time_last_checked = 0

        # Start with first UPF active by default
        self.active_upfs = self.full_upf_pool.iloc[:1].copy().reset_index(drop=True)

    def get_active_upfs(self) -> pd.DataFrame:
        """
        Retrieve the DataFrame of currently active UPFs.

        Returns:
            pd.DataFrame: Active UPFs with original columns.
        """
        return self.active_upfs

    def should_check(self, current_time: float) -> bool:
        """
        Determine if it's time to evaluate scaling decisions.

        Args:
            current_time (float): Current simulation time.

        Returns:
            bool: True if at least 'interval' time has passed since last check.
        """
        return (current_time - self.time_last_checked) >= self.interval

    def update(
        self,
        current_time: float,
        upf_allocations: Dict[int, Dict[int, float]]
    ):
        """
        Perform scaling decision based on active UPFs' CPU utilization.

        - Calculates per-UPF utilization from 'upf_allocations'.
        - If max utilization > scale_out_threshold, adds enough new UPFs.
        - If average utilization < scale_in_threshold, removes one idle UPF.

        Args:
            current_time (float): Current simulation time.
            upf_allocations (Dict[int, Dict[int, float]]): Mapping upf_id -> {pdu_id: cpu_alloc}.
        """
        # Update timestamp for scaling checks
        self.time_last_checked = current_time

        utilizations = []
        # Compute utilization % for each active UPF
        for _, upf in self.active_upfs.iterrows():
            upf_id = upf["upf_id"]
            used = sum(upf_allocations.get(upf_id, {}).values())
            cap = upf["cpu_capacity"]
            utilization = (used / cap) * 100 if cap > 0 else 0
            utilizations.append(utilization)

        # Determine maximum and average utilization
        max_util = max(utilizations) if utilizations else 0
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0

        # Scale out: if peak load exceeds threshold and more UPFs are available
        if (
            max_util > self.scale_out_threshold
            and len(self.active_upfs) < len(self.full_upf_pool)
        ):
            # Identify which UPFs are not currently active
            remaining = self.full_upf_pool[
                ~self.full_upf_pool["upf_id"].isin(self.active_upfs["upf_id"])
            ]
            # Determine how many to add proportional to utilization gap
            gap = max_util - self.scale_out_threshold
            step = 20  # percent per additional UPF
            count = max(1, int(gap / step))
            count = min(count, len(remaining))

            if count > 0:
                to_add = remaining.iloc[:count]
                # Activate new UPFs
                self.active_upfs = pd.concat(
                    [self.active_upfs, to_add], ignore_index=True
                )

        # Scale in: if average load is low and more than one UPF active
        elif avg_util < self.scale_in_threshold and len(self.active_upfs) > 1:
            # Remove the last UPF that has no allocations
            for _, upf in self.active_upfs[::-1].iterrows():
                upf_id = upf["upf_id"]
                if not upf_allocations.get(upf_id):
                    # Deactivate this idle UPF
                    self.active_upfs = self.active_upfs[
                        self.active_upfs["upf_id"] != upf_id
                    ].reset_index(drop=True)
                    break
