import pandas as pd
import matplotlib.pyplot as plt

# Load your datasets (adjust paths if needed)
pdus_df = pd.read_csv('../data/input/pdus.csv')
upfs_df = pd.read_csv('../data/input/upfs.csv')

# Optional: if 'workload_factor' is constant 1.0, else read from upfs_df
workload_factor = 0.00025

# Calculate total available UPF CPU
total_upf_cpu = upfs_df['cpu_capacity'].sum()
print(f"Total available UPF CPU: {total_upf_cpu}")

# Build a time series of required CPU
events = []

for _, row in pdus_df.iterrows():
    events.append((row['start_time'], 'start', row['pdu_id'], row['rate'], row['max_latency']))
    events.append((row['end_time'], 'end', row['pdu_id'], row['rate'], row['max_latency']))

events.sort()

current_pdus = set()
cpu_usage_time = []
time_points = []

for time, event_type, pdu_id, rate, max_latency in events:
    if event_type == 'start':
        current_pdus.add((pdu_id, rate, max_latency))
    elif event_type == 'end':
        current_pdus = {(pid, r, m) for (pid, r, m) in current_pdus if pid != pdu_id}

    total_cpu = sum((workload_factor * r) / m for (pid, r, m) in current_pdus)

    time_points.append(time)
    cpu_usage_time.append(total_cpu)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(time_points, cpu_usage_time, label='Required PDU CPU over time')
plt.axhline(y=total_upf_cpu, color='r', linestyle='--', label='Total UPF CPU')
plt.xlabel('Time')
plt.ylabel('CPU (arbitrary units)')
plt.title('PDU Required CPU vs Available UPF CPU Over Time')
plt.legend()
plt.grid(True)
plt.savefig("cpu_vs_time.png", dpi=300)
print("[INFO] Plot saved to cpu_vs_time.png")

