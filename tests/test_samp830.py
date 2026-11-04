import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from advantage.samp830 import convert_task
from advantage.schema import Episode


def _write_source(root, count=10):
    root = Path(root)
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    records = []
    for episode_id in range(count):
        length = 4 if episode_id % 2 == 0 else 3
        success = episode_id % 2 == 0
        state = np.arange(length * 8, dtype=np.float32).reshape(length, 8) + episode_id
        action = np.zeros((length, 7), dtype=np.float32)
        table = pa.table({
            "state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "actions": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(np.arange(length, dtype=np.float32) * 0.1),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, episode_id, dtype=np.int64)),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        })
        pq.write_table(table, data / f"episode_{episode_id:06d}.parquet")
        records.append({
            "episode_index": episode_id,
            "task_index": 0,
            "suite": "libero_10",
            "task_file": "fake_demo.hdf5",
            "raw_demo": f"demo_{episode_id}",
            "task": "fake task",
            "success": success,
            "length": length,
        })
    (root / "episode_manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return root


class Samp830ConversionTests(unittest.TestCase):
    def test_convert_task_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source")
            output = root / "canonical"
            work = root / "work"
            summary = convert_task(source, 0, output, work, seed=1, task_horizon=4)
            self.assertEqual(summary["episodes"], 10)
            self.assertEqual(sum(summary["split_sizes"].values()), 10)
            self.assertEqual(sum(v["success"] for v in summary["split_outcomes"].values()), 5)
            self.assertEqual(sum(v["failure"] for v in summary["split_outcomes"].values()), 5)
            self.assertTrue((work / "config.json").is_file())
            sidecar = pq.read_table(work / "meta/returns_samp830.parquet")
            self.assertEqual(sidecar.num_rows, 35)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["episodes"]), 10)
            for record in manifest["episodes"]:
                episode = Episode.load(output / record["path"])
                episode_id = int(episode.episode_id.split(":")[-1])
                expected = 1.0 if episode_id % 2 == 0 else 0.0
                self.assertEqual(episode.metadata["success"], episode_id % 2 == 0)
                self.assertAlmostEqual(float(episode.reward.sum()), expected)
                self.assertTrue(episode.terminated[-1])
                self.assertFalse(episode.truncated[-1])
                self.assertEqual(int(episode.remaining_steps[0]), 4)
                self.assertEqual(int(episode.remaining_steps[-1]), 4 - (len(episode) - 1))

    def test_split_warnings_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source", count=2)
            summary = convert_task(source, 0, root / "canonical", root / "work", seed=0, task_horizon=4)
            self.assertEqual(summary["split_seed"], 0)
            self.assertTrue(summary["warnings"])

    def test_invalid_task_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source")
            with self.assertRaises(ValueError):
                convert_task(source, 1, root / "canonical", root / "work", seed=1)

    def test_nonempty_work_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source")
            work = root / "work"
            work.mkdir()
            (work / "unrelated.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_task(source, 0, root / "canonical", work, seed=1, task_horizon=4)

    def test_missing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source")
            (source / "episode_manifest.jsonl").unlink()
            with self.assertRaises(ValueError):
                convert_task(source, 0, root / "canonical", root / "work", seed=1)

    def test_too_small_horizon_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root / "source")
            with self.assertRaises(ValueError):
                convert_task(source, 0, root / "canonical", root / "work", seed=1, task_horizon=2)


if __name__ == "__main__":
    unittest.main()
