import pandas as pd
import matplotlib.pyplot as plt

def plot_active_upfs(results_csv="data/output/results.csv", output_png="results/active_upfs.png"):
    df = pd.read_csv(results_csv)
    starts = df[df["event"] == "START"]
    ends = df[df["event"] == "TERMINATE"]

    events = pd.concat([
        starts.assign(change=1)[["time", "upf_id", "pdu_id", "change"]],
        ends.assign(change=-1)[["time", "upf_id", "pdu_id", "change"]]
    ]).sort_values("time")

    active_upfs = set()
    upf_pdu_map = {}
    timeline = []

    for _, row in events.iterrows():
        upf = row["upf_id"]
        pdu = row["pdu_id"]
        if row["change"] == 1:
            upf_pdu_map.setdefault(upf, set()).add(pdu)
            active_upfs.add(upf)
        else:
            if upf in upf_pdu_map:
                upf_pdu_map[upf].discard(pdu)
                if len(upf_pdu_map[upf]) == 0:
                    active_upfs.discard(upf)

        timeline.append({"time": row["time"], "active_upfs": len(active_upfs)})

    df_plot = pd.DataFrame(timeline).drop_duplicates("time")
    plt.figure(figsize=(20, 10))
    plt.plot(df_plot["time"], df_plot["active_upfs"], label="Active UPFs", color="orange")
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.title("Active UPFs Over Time", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"[INFO] Saved: {output_png}")

if __name__ == "__main__":
    plot_active_upfs()
