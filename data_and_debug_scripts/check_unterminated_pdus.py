import pandas as pd

def check_unterminated_pdus(result_file="data/output/results.csv"):
    df = pd.read_csv(result_file)

    started_pdus = set(df[df["event"] == "START"]["pdu_id"])
    terminated_pdus = set(df[df["event"] == "TERMINATE"]["pdu_id"])

    unterminated = started_pdus - terminated_pdus

    if unterminated:
        print("PDUs started but never terminated:")
        print(sorted(unterminated))
        print(f"Total unterminated PDUs: {len(unterminated)}")
    else:
        print("All started PDUs were properly terminated.")

if __name__ == "__main__":
    check_unterminated_pdus("data/output/results.csv")
