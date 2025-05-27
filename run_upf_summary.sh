#!/bin/bash
set -e

echo "[INFO] Step 1: Computing average UPFs and satisfaction per lambda"
python3 result_scripts/compute_lambda_avg.py

echo "[INFO] Computing average CPU utilization for all lambdas"
python3 result_scripts/compute_lambda_utilization.py

echo "[INFO] Step 2: Generating comparative plots"
python3 result_scripts/plot_summary.py

echo "[DONE] All summaries and plots generated successfully."
