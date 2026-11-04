import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from advantage.cli import main
from advantage.adapters import write_dataset
from advantage.synthetic import generate
from advantage.value import fit_value, ValuePredictor
from advantage.schema import file_hash

class SharedValueTests(unittest.TestCase):
    def test_mc_train_and_shared_checkpoint_scoring(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_dataset(generate(40,77),root/"data")
            checkpoint=root/"shared.pt"
            with patch.object(sys,"argv",["advantage","train",str(root/"data"),"--output",str(checkpoint),"--updates","80"]):
                main()
            predictor=ValuePredictor(checkpoint)
            self.assertEqual(predictor.checkpoint["method"],"mc")
            self.assertIsNone(predictor.checkpoint["n"])
            before=file_hash(checkpoint)
            with patch.object(sys,"argv",["advantage","score",str(root/"data"),"--checkpoint",str(checkpoint),"--output",str(root/"scores.csv")]):
                main()
            with (root/"scores.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual({r["method"] for r in rows},{"n1","n5","n10","n20"})
            self.assertEqual({r["checkpoint_sha256"] for r in rows},{before})
            self.assertEqual(file_hash(checkpoint),before)
            self.assertTrue(all(np.isfinite(float(r["score"])) for r in rows))

    def test_mc_rejects_incomplete_return(self):
        ep=generate(1,10)[0]
        ep.terminated[-1]=False
        ep.truncated[-1]=True
        ep.remaining_steps=ep.remaining_steps+30
        with self.assertRaisesRegex(ValueError,"truncated"):
            fit_value([ep],updates=1)

if __name__ == "__main__": unittest.main()
