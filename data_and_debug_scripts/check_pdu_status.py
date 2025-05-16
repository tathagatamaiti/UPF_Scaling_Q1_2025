import pandas as pd

def check_rejected_pdus(result_file="data/output/results.csv"):
    df = pd.read_csv(result_file)

    started_pdus = set(df[df["event"] == "START"]["pdu_id"])
    terminated_pdus = set(df[df["event"] == "TERMINATE"]["pdu_id"])

    rejected = started_pdus - terminated_pdus

    if rejected:
        print("PDUs rejected:")
        print(sorted(rejected))
        print(f"Total rejected PDUs: {len(rejected)}")
    else:
        print("All PDUs were started and terminated correctly.")

if __name__ == "__main__":
    check_rejected_pdus("data/output/results.csv")
