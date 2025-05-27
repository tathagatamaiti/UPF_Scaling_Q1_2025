import os
import pandas as pd

def compute_active_upf_utilization(results_csv_path: str, upfs_df: pd.DataFrame) -> float:
    try:
        df = pd.read_csv(results_csv_path)
        upf_capacities = dict(zip(upfs_df["upf_id"], upfs_df["cpu_capacity"]))

        start_events = df[df["event"] == "START"]
        end_events = df[df["event"] == "TERMINATE"]

        session_df = pd.merge(
            start_events[["pdu_id", "time", "upf_id", "cpu_allocated"]],
            end_events[["pdu_id", "time"]],
            on="pdu_id",
            suffixes=("_start", "_end")
        )

        time_points = sorted(df["time"].unique())
        utilization_over_time = []

        for t in time_points:
            snapshot = session_df[
                (session_df["time_start"] <= t) &
                (session_df["time_end"] > t)
            ]

            active_upfs = snapshot["upf_id"].unique()
            if len(active_upfs) == 0:
                utilization_over_time.append(0)
                continue

            total_allocated = snapshot["cpu_allocated"].sum()
            total_active_capacity = sum(upf_capacities.get(upf, 1e-9) for upf in active_upfs)

            utilization_percent = (total_allocated / total_active_capacity) * 100
            utilization_percent = min(utilization_percent, 100)
            utilization_over_time.append(utilization_percent)

        return sum(utilization_over_time) / len(utilization_over_time) if utilization_over_time else 0.0
    except Exception as e:
        print(f"[ERROR] Failed for {results_csv_path}: {e}")
        return None

def main():
    base_dir = "data/output"
    upfs_df = pd.read_csv("data/input/upfs.csv")
    modes = ["static", "hpa", "optimizer"]

    for lambda_folder in sorted(os.listdir(base_dir)):
        lambda_path = os.path.join(base_dir, lambda_folder)
        if not os.path.isdir(lambda_path) or not lambda_folder.startswith("lambda_"):
            continue

        mode_utilizations = {mode: [] for mode in modes}

        for seed_folder in os.listdir(lambda_path):
            seed_path = os.path.join(lambda_path, seed_folder)
            if not os.path.isdir(seed_path) or not seed_folder.startswith("seed_"):
                continue

            for mode in modes:
                file_path = os.path.join(seed_path, f"results_{mode}.csv")
                if os.path.exists(file_path):
                    util = compute_active_upf_utilization(file_path, upfs_df)
                    if util is not None:
                        mode_utilizations[mode].append(util)

        summary = pd.DataFrame([{
            "avg_static_utilization_active_upfs (%)":
                sum(mode_utilizations["static"]) / len(mode_utilizations["static"]) if mode_utilizations["static"] else None,
            "avg_hpa_utilization_active_upfs (%)":
                sum(mode_utilizations["hpa"]) / len(mode_utilizations["hpa"]) if mode_utilizations["hpa"] else None,
            "avg_optimizer_utilization_active_upfs (%)":
                sum(mode_utilizations["optimizer"]) / len(mode_utilizations["optimizer"]) if mode_utilizations["optimizer"] else None
        }])

        output_file = os.path.join(lambda_path, "avg_utilization_active_upfs.csv")
        summary.to_csv(output_file, index=False)
        print(f"[INFO] Saved: {output_file}")

if __name__ == "__main__":
    main()
