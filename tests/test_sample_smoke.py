import csv
import json
import tempfile
import unittest
from pathlib import Path
from advantage.sample_smoke import run

SAMPLES=Path(__file__).resolve().parents[1]/"samples"

@unittest.skipUnless((SAMPLES/"manifest.jsonl").exists(),"real sample fixtures unavailable")
class RealSampleSmokeTests(unittest.TestCase):
    def test_real_samples_convert_train_reload_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"smoke"
            report=run(SAMPLES,output,updates=2,device="cpu")
            self.assertEqual(report["status"],"passed")
            self.assertEqual(report["scope"],"same_episode_smoke_only")
            self.assertEqual(report["raw_samples"],4)
            self.assertEqual(report["transitions"],1257)
            self.assertEqual(report["score_rows"],6285)
            self.assertEqual(sum(a["success"] for a in report["audits"]),2)
            self.assertTrue(all(not t["normal_train_split_available"] for t in report["tasks"]))
            with (output/"scores.csv").open() as f:rows=list(csv.DictReader(f))
            self.assertEqual({r["method"] for r in rows},{"td_n1","td_n5","td_n10","td_n20","mc_q_minus_shared_v"})
            self.assertEqual({r["scope"] for r in rows},{"same_episode_smoke_only"})
            self.assertTrue((output/"report.json").exists())
            with self.assertRaisesRegex(ValueError,"new empty output"):
                run(SAMPLES,output,updates=2)

if __name__ == "__main__":unittest.main()
