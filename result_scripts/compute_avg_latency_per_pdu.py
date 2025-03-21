import pandas as pd

def compute_normalized_avg_latency(results_csv="data/output/results.csv", output_csv="data/output/pdu_latency_summary.csv"):
    df = pd.read_csv(results_csv)

    df = df[df["event"].isin(["START", "REBALANCE"])]

    avg_df = df.groupby("pdu_id").agg({
        "observed_latency": "mean",
        "required_max_latency": "first"
    }).reset_index()

    avg_df["normalized_avg_latency"] = avg_df["observed_latency"] / avg_df["required_max_latency"]

    avg_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved normalized average latency per PDU to {output_csv}")

if __name__ == "__main__":
    compute_normalized_avg_latency()
