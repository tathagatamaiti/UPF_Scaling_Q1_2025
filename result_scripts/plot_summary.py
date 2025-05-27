import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_context("talk", font_scale=1.3)
sns.set_style("whitegrid")
palette = sns.color_palette("colorblind")

mu = 0.02
root_dir = "data/output"
records = []

for folder in sorted(os.listdir(root_dir)):
    lambda_path = os.path.join(root_dir, folder)
    summary_path = os.path.join(lambda_path, "average_upfs_summary.csv")
    util_path = os.path.join(lambda_path, "avg_utilization_active_upfs.csv")

    if not os.path.isfile(summary_path):
        continue

    try:
        df = pd.read_csv(summary_path)
        lambda_val = float(folder.replace("lambda_", ""))
        record = df.iloc[0].to_dict()

        if os.path.isfile(util_path):
            util_df = pd.read_csv(util_path)
            record.update(util_df.iloc[0].to_dict())

        record["lambda/mu"] = lambda_val / mu
        records.append(record)
    except Exception as e:
        print(f"[WARNING] Skipping {folder}: {e}")

summary_df = pd.DataFrame(records).sort_values(by="lambda/mu")

def fancy_plot(x, ys, yerrs, labels, ylabel, title, filename):
    plt.figure(figsize=(12, 7))
    markers = ["o", "s", "D"]
    for i, y in enumerate(ys):
        plt.errorbar(
            summary_df[x],
            summary_df[y],
            yerr=summary_df.get(yerrs[i]),
            label=labels[i],
            marker=markers[i],
            capsize=3,
            linewidth=2,
            markersize=5
        )
    plt.xlabel(r"$\lambda / \mu$", fontsize=18, labelpad=10)
    plt.ylabel(ylabel, fontsize=18, labelpad=10)
    plt.title(title, fontsize=20, fontweight="bold", pad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, filename), dpi=300)
    plt.show()

fancy_plot(
    x="lambda/mu",
    ys=["avg_static_upfs", "avg_hpa_upfs", "avg_optimizer_upfs"],
    yerrs=["std_static_upfs", "std_hpa_upfs", "std_optimizer_upfs"],
    labels=["Static", "HPA", "Optimizer"],
    ylabel="Average UPFs Used",
    title="Average Number of UPFs vs λ / μ",
    filename="plot_upfs_vs_lambda_mu.png"
)

fancy_plot(
    x="lambda/mu",
    ys=["avg_static_rejection (%)", "avg_hpa_rejection (%)", "avg_optimizer_rejection (%)"],
    yerrs=["std_static_rejection (%)", "std_hpa_rejection (%)", "std_optimizer_rejection (%)"],
    labels=["Static", "HPA", "Optimizer"],
    ylabel="Rejection Rate (%)",
    title="Rejection Rate vs λ / μ",
    filename="plot_rejection_vs_lambda_mu.png"
)

fancy_plot(
    x="lambda/mu",
    ys=["avg_static_satisfaction (%)", "avg_hpa_satisfaction (%)", "avg_optimizer_satisfaction (%)"],
    yerrs=["std_static_satisfaction (%)", "std_hpa_satisfaction (%)", "std_optimizer_satisfaction (%)"],
    labels=["Static", "HPA", "Optimizer"],
    ylabel="Satisfaction Rate (%)",
    title="Satisfaction Rate vs λ / μ",
    filename="plot_satisfaction_vs_lambda_mu.png"
)

if "avg_static_utilization_active_upfs (%)" in summary_df.columns:
    plt.figure(figsize=(12, 7))
    plt.plot(summary_df["lambda/mu"], summary_df["avg_static_utilization_active_upfs (%)"],
             label="Static", marker="o", linewidth=2, markersize=5)
    plt.plot(summary_df["lambda/mu"], summary_df["avg_hpa_utilization_active_upfs (%)"],
             label="HPA", marker="s", linewidth=2, markersize=5)
    plt.plot(summary_df["lambda/mu"], summary_df["avg_optimizer_utilization_active_upfs (%)"],
             label="Optimizer", marker="D", linewidth=2, markersize=5)
    plt.xlabel(r"$\lambda / \mu$", fontsize=18)
    plt.ylabel("Avg. Cluster Utilization (%)", fontsize=18)
    plt.title("Average Cluster Utilization vs λ / μ", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "plot_utilization_vs_lambda_mu.png"), dpi=300)
    plt.show()
