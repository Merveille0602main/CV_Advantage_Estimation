import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa=None
from advantage.adapters import load_rlinf, write_dataset


@unittest.skipIf(pa is None,"install rlinf extra to test Parquet integration")
class RLinfTests(unittest.TestCase):
    def fixture(self, root):
        (root/"data/chunk-000").mkdir(parents=True)
        (root/"meta").mkdir()
        rows=[{"episode_index":e,"frame_index":i,"observation.state":[float(i),2.],
               "action":[.1,.2],"terminated":i==2} for e in range(12) for i in range(3)]
        # Deliberately reversed file row order: alignment must use keys, not position.
        pq.write_table(pa.Table.from_pylist(rows[::-1]),root/"data/chunk-000/part.parquet")
        rewards=[{"episode_index":e,"frame_index":i,"reward":float(i==2),
                  "return":999.,"advantage":True} for e in range(12) for i in range(3)]
        pq.write_table(pa.Table.from_pylist(rewards),root/"meta/returns_test.parquet")
        return dict(source_id="fixture-v1",policy_id="fixed-v1",task_id="task0",task_horizon=3,
          state_columns=["observation.state"],action_column="action",reward_column="reward",
          reward_sidecar="meta/returns_test.parquet",terminal_column="terminated",
          rows_are_primitive_transitions=True)

    def test_sidecar_join_roundtrip_and_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg=self.fixture(root); eps=load_rlinf(root,cfg)
            self.assertEqual(len(eps),12)
            np.testing.assert_array_equal(eps[0].reward,[0,0,1])
            np.testing.assert_array_equal(eps[0].next_obs[:-1],eps[0].obs[1:])
            manifest=write_dataset(eps,root/"canonical")
            self.assertTrue(all(manifest["split"].values()))
            with self.assertRaises(ValueError): write_dataset(eps,root/"canonical")

    def test_missing_reward_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root)
            cfg["reward_column"]="not_a_reward"
            with self.assertRaises(ValueError): load_rlinf(root,cfg)

    def test_chunk_semantics_must_be_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root);cfg["rows_are_primitive_transitions"]=False
            with self.assertRaises(ValueError): load_rlinf(root,cfg)

    def test_truncation_without_next_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root)
            rows=[{"episode_index":0,"frame_index":i,"observation.state":[float(i)],
                   "action":[0.],"terminated":False,"truncated":i==2} for i in range(3)]
            pq.write_table(pa.Table.from_pylist(rows),root/"data/chunk-000/part.parquet")
            cfg.update(task_horizon=10,truncated_column="truncated")
            with self.assertRaisesRegex(ValueError,"real final next observation"): load_rlinf(root,cfg)


if __name__ == "__main__": unittest.main()
