import os
import pandas as pd

def compute_lambda_averages(lambda_dir):
    static_upfs, hpa_upfs, opt_upfs = [], [], []
    static_rejection, hpa_rejection, opt_rejection = [], [], []
    static_satisfaction, hpa_satisfaction, opt_satisfaction = [], [], []

    for seed_folder in os.listdir(lambda_dir):
        seed_path = os.path.join(lambda_dir, seed_folder)
        if not os.path.isdir(seed_path) or not seed_folder.startswith("seed_"):
            continue

        try:
            for mode in ["static", "hpa", "optimizer"]:
                file_path = os.path.join(seed_path, f"summary_{mode}.csv")
                df = pd.read_csv(file_path)

                used_upfs = df["used_upfs"].iloc[0]
                total_pdus = df["total_pdus"].iloc[0]
                rejected_pdus = df["rejected_pdus"].iloc[0]

                rejection = (rejected_pdus / total_pdus) * 100 if total_pdus else 0
                satisfaction = 100 - rejection

                if mode == "static":
                    static_upfs.append(used_upfs)
                    static_rejection.append(rejection)
                    static_satisfaction.append(satisfaction)

                elif mode == "hpa":
                    hpa_upfs.append(used_upfs)
                    hpa_rejection.append(rejection)
                    hpa_satisfaction.append(satisfaction)

                elif mode == "optimizer":
                    opt_upfs.append(used_upfs)
                    opt_rejection.append(rejection)
                    opt_satisfaction.append(satisfaction)

        except Exception as e:
            print(f"[WARNING] Skipping {seed_path}: {e}")

    return pd.DataFrame([{
        "avg_static_upfs": pd.Series(static_upfs).mean() if static_upfs else None,
        "std_static_upfs": pd.Series(static_upfs).std() if static_upfs else None,
        "avg_static_rejection (%)": pd.Series(static_rejection).mean() if static_rejection else None,
        "std_static_rejection (%)": pd.Series(static_rejection).std() if static_rejection else None,
        "avg_static_satisfaction (%)": pd.Series(static_satisfaction).mean() if static_satisfaction else None,
        "std_static_satisfaction (%)": pd.Series(static_satisfaction).std() if static_satisfaction else None,

        "avg_hpa_upfs": pd.Series(hpa_upfs).mean() if hpa_upfs else None,
        "std_hpa_upfs": pd.Series(hpa_upfs).std() if hpa_upfs else None,
        "avg_hpa_rejection (%)": pd.Series(hpa_rejection).mean() if hpa_rejection else None,
        "std_hpa_rejection (%)": pd.Series(hpa_rejection).std() if hpa_rejection else None,
        "avg_hpa_satisfaction (%)": pd.Series(hpa_satisfaction).mean() if hpa_satisfaction else None,
        "std_hpa_satisfaction (%)": pd.Series(hpa_satisfaction).std() if hpa_satisfaction else None,

        "avg_optimizer_upfs": pd.Series(opt_upfs).mean() if opt_upfs else None,
        "std_optimizer_upfs": pd.Series(opt_upfs).std() if opt_upfs else None,
        "avg_optimizer_rejection (%)": pd.Series(opt_rejection).mean() if opt_rejection else None,
        "std_optimizer_rejection (%)": pd.Series(opt_rejection).std() if opt_rejection else None,
        "avg_optimizer_satisfaction (%)": pd.Series(opt_satisfaction).mean() if opt_satisfaction else None,
        "std_optimizer_satisfaction (%)": pd.Series(opt_satisfaction).std() if opt_satisfaction else None
    }])

def main():
    base_output = "data/output"
    for folder in sorted(os.listdir(base_output)):
        lambda_path = os.path.join(base_output, folder)
        if not os.path.isdir(lambda_path) or not folder.startswith("lambda_"):
            continue

        result_df = compute_lambda_averages(lambda_path)
        output_file = os.path.join(lambda_path, "average_upfs_summary.csv")
        result_df.to_csv(output_file, index=False)
        print(f"✅ Saved summary for {folder} to {output_file}")

if __name__ == "__main__":
    main()
