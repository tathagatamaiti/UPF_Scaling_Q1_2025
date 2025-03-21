import pandas as pd
import matplotlib.pyplot as plt

def plot_active_pdus(results_csv="data/output/results.csv", output_png="results/active_pdus.png"):
    df = pd.read_csv(results_csv)
    starts = df[df["event"] == "START"]
    ends = df[df["event"] == "TERMINATE"]

    events = pd.concat([
        starts.assign(change=1)[["time", "pdu_id", "change"]],
        ends.assign(change=-1)[["time", "pdu_id", "change"]]
    ]).sort_values("time")

    active_pdus = 0
    timeline = []
    for _, row in events.iterrows():
        active_pdus += row["change"]
        timeline.append({"time": row["time"], "active_pdus": active_pdus})

    df_plot = pd.DataFrame(timeline).drop_duplicates("time")
    plt.figure(figsize=(20, 10))
    plt.plot(df_plot["time"], df_plot["active_pdus"], label="Active PDUs")
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.title("Active PDUs Over Time", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"[INFO] Saved: {output_png}")

if __name__ == "__main__":
    plot_active_pdus()
