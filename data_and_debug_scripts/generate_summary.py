import pandas as pd

def generate_summary(input_csv="results.csv", output_csv="summary.csv"):
    df = pd.read_csv(input_csv)

    df_start = df[df["event"] == "START"]

    total_pdus = len(df_start)
    satisfied = (df_start["status"] == "SATISFIED").sum()
    unsatisfied = (df_start["status"] == "UNSATISFIED").sum()
    rejected = (df_start["status"] == "REJECTED").sum()
    used_upfs = df_start[df_start["status"] != "REJECTED"]["upf_id"].nunique()

    summary_df = pd.DataFrame([{
        "total_pdus": total_pdus,
        "satisfied_pdus": satisfied,
        "unsatisfied_pdus": unsatisfied,
        "rejected_pdus": rejected,
        "used_upfs": used_upfs,
        "satisfaction_ratio (%)": round((satisfied / total_pdus) * 100, 2) if total_pdus > 0 else 0
    }])

    summary_df.to_csv(output_csv, index=False)
    print(f"[INFO] Summary saved to {output_csv}")
    print(summary_df)
