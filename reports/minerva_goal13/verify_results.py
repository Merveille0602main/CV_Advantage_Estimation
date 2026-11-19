"""Recompute the published metrics from frozen weights and saved evaluation branches."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "scripts"), str(HERE.parents[1] / "src")]
import branch_metrics


def compare_values(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            compare_values(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_values(a, e)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        assert actual == expected, (actual, expected)


def main():
    branch_metrics.SOURCE = HERE / "20261119_run01"
    source = HERE / "20261119_branch01"
    expected = json.loads((source / "results.json").read_text())
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        for filename in ["protocol.json", *expected["simulation_files"]]:
            shutil.copyfile(source / filename, out / filename)
        with contextlib.redirect_stdout(io.StringIO()):
            branch_metrics.main(out)
        for filename in ("results.json", "reference_pairs.json", "action_values_and_scores.json"):
            compare_values(json.loads((out / filename).read_text()), json.loads((source / filename).read_text()))
    print(f"Verified {expected['eligible_pairs']}/{expected['pairs']} action pairs and all published Q metrics.")


if __name__ == "__main__":
    main()
