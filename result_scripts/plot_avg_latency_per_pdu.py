import pandas as pd
import matplotlib.pyplot as plt

def plot_latency(input_csv="data/output/pdu_latency_summary.csv", output_png="results/normalized_avg_latency_per_pdu.png"):
    df = pd.read_csv(input_csv)
    df_sorted = df.sort_values("pdu_id")

    plt.figure(figsize=(20, 10))
    plt.plot(df_sorted["pdu_id"], df_sorted["normalized_avg_latency"], marker='o')
    plt.xlabel("PDU ID", fontsize=18)
    plt.ylabel("Normalized Avg Latency (observed / max)", fontsize=18)
    plt.title("Normalized Average Latency per PDU", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"[INFO] Plot saved to {output_png}")

if __name__ == "__main__":
    plot_latency()
