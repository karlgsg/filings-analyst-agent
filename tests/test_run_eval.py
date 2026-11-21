import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "evals"))

from run_eval import load_checkpoint


def test_load_checkpoint_reads_saved_rows(tmp_path):
    path = tmp_path / "run.partial.jsonl"
    assert load_checkpoint(path) == []  # no crashed run yet
    path.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n\n")
    assert [r["id"] for r in load_checkpoint(path)] == ["a", "b"]
