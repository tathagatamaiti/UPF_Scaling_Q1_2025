import matplotlib.pyplot as plt
import pandas as pd

def plot_cpu_utilization(csv_file: str, upfs_file: str, save_path=None):
    results_df = pd.read_csv(csv_file)
    upfs_df = pd.read_csv(upfs_file)

    cpu_capacity = {row['upf_id']: row['cpu_capacity'] for _, row in upfs_df.iterrows()}
    upf_ids = list(cpu_capacity.keys())

    # Only START events are used to track allocation
    start_events = results_df[results_df["event"] == "START"]
    end_events = results_df[results_df["event"] == "TERMINATE"]

    # Merge start and end times
    session_df = pd.merge(start_events, end_events, on="pdu_id", suffixes=("_start", "_end"))
    time_points = sorted(set(results_df["time"]))

    records = []

    for t in time_points:
        snapshot = {"time": t}
        for upf_id in upf_ids:
            active = session_df[
                (session_df["upf_id_start"] == upf_id) &
                (session_df["time_start"] <= t) &
                (session_df["time_end"] > t)
            ]
            total_cpu = active["cpu_allocated_start"].sum()
            utilization = (total_cpu / cpu_capacity[upf_id]) * 100
            snapshot[f"UPF_{upf_id}"] = utilization
        records.append(snapshot)

    df_plot = pd.DataFrame(records).set_index("time")

    # Plotting
    plt.figure(figsize=(20, 10))
    for col in df_plot.columns:
        plt.plot(df_plot.index, df_plot[col], label=col)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("CPU Utilization (%)", fontsize=18)
    plt.title("UPF CPU Utilization Over Time", fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()
