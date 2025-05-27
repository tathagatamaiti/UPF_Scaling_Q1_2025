set -e

pip install --upgrade pip
pip install pandas matplotlib typing numpy pyomo

lambdas=(15)
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

        echo "[INFO] Running unit tests for lambda=${lambda}, seed=${seed}"
        python3 -m unittest simulation_scripts/test_simulator.py

        for mode in static hpa optimizer; do
            echo "[INFO] Running simulation (${mode}) for lambda=${lambda}, seed=${seed}"
            python3 <<EOF
from simulation_scripts.simulator import PDUScheduler
scheduler = PDUScheduler("data/input/pdus.csv", "data/input/upfs.csv", mode="${mode}")
scheduler.run()
scheduler.export_results("${output_dir}/results_${mode}.csv")
EOF
        done

        for mode in static hpa optimizer; do
            echo "[INFO] Generating summary for ${mode} (lambda=${lambda}, seed=${seed})"
            python3 -c "
from data_and_debug_scripts.generate_summary import generate_summary
generate_summary('${output_dir}/results_${mode}.csv', '${output_dir}/summary_${mode}.csv')
"
        done

        echo "[INFO] Checking PDU status for lambda=${lambda}, seed=${seed}"
        python3 data_and_debug_scripts/check_pdu_status.py

        echo "[SUCCESS] Simulation completed for lambda=${lambda}, seed=${seed}"
    done
done

echo "[SUCCESS] All simulations complete. Results and plots saved in /data/output"
