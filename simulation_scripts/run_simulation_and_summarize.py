import sys
import os
import pandas as pd
from simulator import (
    PDUScheduler,
    compute_active_upf_utilization,
    compute_average_normalized_latency,
    generate_summary_from_results
)

def run_all_modes(pdus_file: str, upfs_file: str, output_dir: str):
    modes = ["static", "hpa", "optimizer"]

    utilization_rows = []
    summary_rows = []

    for mode in modes:
        print(f"\n[INFO] Running simulation in mode: {mode.upper()}")
        scheduler = PDUScheduler(pdus_file, upfs_file, mode=mode)
        scheduler.run()
        results_df = scheduler.get_results_df()

        utilization = compute_active_upf_utilization(results_df, scheduler.upfs_df)
        normalized_latency = compute_average_normalized_latency(results_df)
        summary_df = generate_summary_from_results(results_df, normalized_latency)

        utilization_rows.append({
            "mode": mode,
            "avg_active_upf_utilization (%)": round(utilization, 2)
        })

        summary_row = summary_df.iloc[0].to_dict()
        summary_row["mode"] = mode
        summary_rows.append(summary_row)

        results_output_path = os.path.join(output_dir, f"{mode}_results.csv")
        results_df.to_csv(results_output_path, index=False)

    pd.DataFrame(utilization_rows).to_csv(os.path.join(output_dir, "average_utilization.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    print("\n[INFO] Simulation completed. Results saved to:", output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_simulation_and_summarize.py <pdus_file.csv> <upfs_file.csv> <output_dir>")
        sys.exit(1)

    pdus_file = sys.argv[1]
    upfs_file = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    run_all_modes(pdus_file, upfs_file, output_dir)
