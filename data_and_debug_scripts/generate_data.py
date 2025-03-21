import pandas as pd
import numpy as np

def generate_pdu_data(
    end_time_limit=100, lam=1.0, mu=0.02,
    min_latency_range=(0.25, 0.5),
    max_latency_range=(0.1, 0.5),
    rate_range=(100000, 200000),
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    inter_arrivals = []
    t = 0
    while t < end_time_limit:
        iat = np.random.exponential(lam)
        t += iat
        if t < end_time_limit:
            inter_arrivals.append(iat)

    num_pdus = len(inter_arrivals)
    start_times = np.cumsum(inter_arrivals)
    durations = np.random.exponential(1 / mu, num_pdus)
    end_times = start_times + durations
    min_latencies = np.random.uniform(*min_latency_range, num_pdus)
    max_latencies = min_latencies + np.random.uniform(*max_latency_range, num_pdus)
    rates = np.random.uniform(*rate_range, num_pdus)

    pdus_df = pd.DataFrame({
        "pdu_id": range(1, num_pdus + 1),
        "start_time": np.round(start_times, 2),
        "end_time": np.round(end_times, 2),
        "min_latency": np.round(min_latencies, 2),
        "max_latency": np.round(max_latencies, 2),
        "rate": np.round(rates, 2)
    })

    pdus_df.to_csv("data/input/pdus.csv", index=False)
    print(f"[INFO] pdus.csv generated with {num_pdus} PDUs (up to time {end_time_limit})")


def generate_upf_data(
    num_upfs=5,
    workload=0.00025,
    capacity=200,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    upfs_df = pd.DataFrame({
        "upf_id": range(0, 0 + num_upfs),
        "workload_factor": [workload] * num_upfs,
        "cpu_capacity": [capacity] * num_upfs
    })

    upfs_df.to_csv("data/input/upfs.csv", index=False)
    print(f"[INFO] upfs.csv generated with {num_upfs} UPFs (workload={workload}, capacity={capacity})")

if __name__ == "__main__":
    generate_pdu_data(end_time_limit=100, lam=1, mu=0.02, seed=42)
    generate_upf_data(num_upfs=5, seed=42)
