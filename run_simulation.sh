set -e

pip install --upgrade pip
pip install pandas matplotlib typing numpy

echo "[INFO] Generating PDU and UPF data"
python3 data_and_debug_scripts/generate_data.py

echo "[INFO] Running unit tests"
python3 -m unittest simulation_scripts/test_simulator.py

echo "[INFO] Running simulator and exporting results"
python3 <<EOF
from simulation_scripts.simulator import PDUScheduler
from result_scripts.plot_utilization import plot_cpu_utilization

scheduler = PDUScheduler("data/input/pdus.csv", "data/input/upfs.csv")
scheduler.run()
scheduler.export_results("data/output/results.csv")
plot_cpu_utilization("data/output/results.csv", "data/input/upfs.csv", save_path="results/cpu_utilization.png")
EOF

echo "[INFO] Simulation complete."
echo "[INFO] Results saved to results.csv"
echo "[INFO] CPU utilization plot saved to cpu_utilization.png"
echo "[INFO] Computing and plotting average PDU latency"
python3 result_scripts/compute_avg_latency_per_pdu.py
python3 result_scripts/plot_avg_latency_per_pdu.py
echo "[INFO] Plotting active PDUs and UPFs"
python3 result_scripts/plot_active_pdus.py
python3 result_scripts/plot_active_upfs.py

echo "[INFO] Generating summary report"
python3 -c "from data_and_debug_scripts.generate_summary import generate_summary; generate_summary('data/output/results.csv', 'data/output/summary.csv')"

echo "[INFO] Check if all PDUs are properly terminated"
python3 data_and_debug_scripts/check_unterminated_pdus.py

