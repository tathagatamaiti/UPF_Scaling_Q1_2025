import pandas as pd

def generate_summary(input_csv="results.csv", output_csv="summary.csv"):
    df = pd.read_csv(input_csv)

    df_start = df[df["event"] == "START"].drop_duplicates(subset="pdu_id")

    total_pdus = df_start["pdu_id"].nunique()
    satisfied = (df_start["status"] == "SATISFIED").sum()
    unsatisfied = (df_start["status"] == "UNSATISFIED").sum()

    summary_df = pd.DataFrame([{
        "total_pdus": total_pdus,
        "satisfied_pdus": satisfied,
        "unsatisfied_pdus": unsatisfied,
        "satisfaction_ratio (%)": round((satisfied / total_pdus) * 100, 2) if total_pdus > 0 else 0
    }])

    summary_df.to_csv(output_csv, index=False)
    print(f"[INFO] Summary saved to {output_csv}")
    print(summary_df)
