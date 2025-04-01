set -e

pip install --upgrade pip
pip install pandas matplotlib typing numpy

echo "[INFO] Generating PDU and UPF data"
python3 data_and_debug_scripts/generate_data.py

echo "[INFO] Running unit tests"
python3 -m unittest simulation_scripts/test_simulator.py

echo "[INFO] Running simulation WITHOUT HPA"
python3 <<EOF
from simulation_scripts.simulator import PDUScheduler

scheduler = PDUScheduler("data/input/pdus.csv", "data/input/upfs.csv", use_hpa=False)
scheduler.run()
scheduler.export_results("data/output/results_no_hpa.csv")
EOF

echo "[INFO] Running simulation WITH HPA"
python3 <<EOF
from simulation_scripts.simulator import PDUScheduler

scheduler = PDUScheduler("data/input/pdus.csv", "data/input/upfs.csv", use_hpa=True)
scheduler.run()
scheduler.export_results("data/output/results_hpa.csv")
EOF

echo "[INFO] Generating summary reports"
python3 -c "from data_and_debug_scripts.generate_summary import generate_summary; \
    generate_summary('data/output/results_no_hpa.csv', 'data/output/summary_no_hpa.csv')"
python3 -c "from data_and_debug_scripts.generate_summary import generate_summary; \
    generate_summary('data/output/results_hpa.csv', 'data/output/summary_hpa.csv')"

echo "[INFO] Checking PDU status"
python3 data_and_debug_scripts/check_pdu_status.py

echo "[SUCCESS] All simulations complete. Results and plots saved in /data/output and /results"
