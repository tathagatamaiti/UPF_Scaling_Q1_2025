#!/bin/bash
set -e

pip install --upgrade pip
pip install pandas matplotlib typing numpy pyomo

lambdas=(0.1 0.3 0.5 0.7 1 3 5 7 10 15 20)
seeds=(1000000007 1000000009 1000000021 1000000033 1000000087 1000000093 1000000103 1000000123 1000000181 1000000207)

for lambda in "${lambdas[@]}"; do
    for seed in "${seeds[@]}"; do

        echo "[INFO] Running for lambda=${lambda}, seed=${seed}"

        python3 -c "
import numpy as np
from data_and_debug_scripts.generate_data import generate_pdu_data, generate_upf_data
generate_pdu_data(end_time_limit=3600, lam=${lambda}, mu=0.02, seed=${seed})
generate_upf_data(num_upfs=100, seed=${seed})
print('[INFO] Data generated for lambda=${lambda}, seed=${seed}')
"

        output_dir="data/output/lambda_${lambda}/seed_${seed}"
        mkdir -p "$output_dir"

        echo "[INFO] Running simulation and summarization for lambda=${lambda}, seed=${seed}"
        python3 simulation_scripts/run_simulation_and_summarize.py \
            data/input/pdus.csv \
            data/input/upfs.csv \
            "$output_dir"

        echo "[SUCCESS] Completed lambda=${lambda}, seed=${seed}"
    done
done

echo "[SUCCESS] All simulations complete. Results saved in data/output/"


