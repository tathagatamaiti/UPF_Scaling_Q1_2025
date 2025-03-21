import unittest
import pandas as pd
import os
from simulation_scripts.simulator import PDUScheduler

class TestPDUScheduler(unittest.TestCase):
    def setUp(self):
        self.pdus_file = "test_pdus.csv"
        self.upfs_file = "test_upfs.csv"

        pdus = pd.DataFrame({
            "pdu_id": [1, 2],
            "start_time": [0, 1],
            "end_time": [5, 6],
            "min_latency": [1, 1],
            "max_latency": [10, 10],
            "rate": [5, 10]
        })

        upfs = pd.DataFrame({
            "upf_id": [101],
            "workload_factor": [1.0],
            "cpu_capacity": [100.0]
        })

        pdus.to_csv(self.pdus_file, index=False)
        upfs.to_csv(self.upfs_file, index=False)

    def tearDown(self):
        os.remove(self.pdus_file)
        os.remove(self.upfs_file)
        if os.path.exists("allocation_results.csv"):
            os.remove("allocation_results.csv")

    def test_allocation_and_logging(self):
        scheduler = PDUScheduler(self.pdus_file, self.upfs_file)
        scheduler.run()
        scheduler.export_results()

        df = pd.read_csv("allocation_results.csv")
        first_starts = df[df["event"] == "START"].drop_duplicates(subset="pdu_id")
        terminations = df[df["event"] == "TERMINATE"]
        self.assertEqual(len(first_starts), 2)
        self.assertEqual(len(terminations), 2)
        self.assertTrue((df["event"] == "START").any())
        self.assertTrue((df["event"] == "TERMINATE").any())
        self.assertIn("cpu_allocated", df.columns)
        self.assertIn("observed_latency", df.columns)

    def test_satisfaction_status(self):
        scheduler = PDUScheduler(self.pdus_file, self.upfs_file)
        scheduler.run()
        satisfied = [entry['status'] for entry in scheduler.result_log if entry['event'] == "START"]
        self.assertTrue(all(status in ["SATISFIED", "UNSATISFIED"] for status in satisfied))

    def test_no_active_pdus_left(self):
        scheduler = PDUScheduler(self.pdus_file, self.upfs_file)
        scheduler.run()
        self.assertEqual(len(scheduler.active_pdus), 0)

if __name__ == '__main__':
    unittest.main()
