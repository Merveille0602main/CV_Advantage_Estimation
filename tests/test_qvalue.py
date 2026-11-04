import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import torch
from advantage.synthetic import generate
from advantage.qvalue import fit_q,QPredictor,qv_scores
from advantage.value import fit_value,ValuePredictor
from advantage.evaluation import ranking_metrics
from advantage.adapters import write_dataset
from advantage.cli import main,read_split

class RankingTests(unittest.TestCase):
    def test_independent_reference_and_ties(self):
        result=ranking_metrics([[1,2],[1,1],[2,1]],[[0,1],[0,1],[0,1]],0)
        self.assertAlmostEqual(result["ranking_accuracy"],.5)
        self.assertEqual(result["valid_pairs"],3)
    def test_uncertain_labels_preserve_coverage(self):
        result=ranking_metrics([[1,2],[1,2]],[[1,1.001],[1,2]],.005)
        self.assertEqual(result["pair_coverage"],.5)
        self.assertEqual(result["evaluable_anchors"],1)
    def test_no_distinguishable_labels(self):
        self.assertIsNone(ranking_metrics([[0,1]],[[1,1]])["ranking_accuracy"])

class QValueTests(unittest.TestCase):
    def test_training_candidate_scoring_and_protocol_checks(self):
        torch.set_num_threads(1)
        train=generate(30,4)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);q=root/"q.pt";v=root/"v.pt"
            torch.save(fit_q(train,updates=80),q)
            torch.save(fit_value(train,updates=80),v)
            qp,vp=QPredictor(q),ValuePredictor(v)
            ep=generate(1,90)[0]
            scores=qv_scores(qp,vp,ep)
            self.assertEqual(scores.shape,(len(ep),))
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(np.isfinite(qp.predict(ep,np.ones_like(ep.action))).all())
            with self.assertRaises(ValueError):qp.predict(ep,np.zeros((len(ep),2)))
            qp.checkpoint["gamma"]=.5
            with self.assertRaisesRegex(ValueError,"gamma"):qv_scores(qp,vp,ep)
    def test_q_rejects_truncation_and_policy_mixture(self):
        eps=generate(2,9);eps[1].policy_id="other"
        with self.assertRaises(ValueError):fit_q(eps,updates=1)
        ep=eps[0];ep.terminated[-1]=False;ep.truncated[-1]=True;ep.remaining_steps+=30
        with self.assertRaisesRegex(ValueError,"truncated"):fit_q([ep],updates=1)
    def test_cli_and_family_leakage_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=write_dataset(generate(40,8),root/"data")
            for command in ("train","train-q"):
                output=root/(command+".pt")
                with patch.object(sys,"argv",["advantage",command,str(root/"data"),"--output",str(output),"--updates","20"]):main()
            with patch.object(sys,"argv",["advantage","score-qv",str(root/"data"),"--q-checkpoint",str(root/"train-q.pt"),"--v-checkpoint",str(root/"train.pt"),"--output",str(root/"scores.csv")]):main()
            with (root/"scores.csv").open() as f:rows=list(csv.DictReader(f))
            self.assertTrue(rows)
            self.assertEqual({r["method"] for r in rows},{"mc_q_minus_shared_v"})
            # Corrupt manifest family assignment; reader must reject leakage before fitting.
            train_id=manifest["split"]["train"][0];test_id=manifest["split"]["test"][0]
            train_record=next(r for r in manifest["episodes"] if r["episode_id"]==train_id)
            test_record=next(r for r in manifest["episodes"] if r["episode_id"]==test_id)
            train_record["parent_id"]=test_record["parent_id"]
            (root/"data/manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):read_split(root/"data","test")

if __name__ == "__main__":unittest.main()
