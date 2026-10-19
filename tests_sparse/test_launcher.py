import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "Perplexity_LongBench"))
from run_sparse_longbench import validate_manifest, save_new


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT.parent / "Perplexity_LongBench/experiments/celerity_all_4k_a6000_bf16/manifest.json"
        self.original = self.path.read_bytes()
        self.manifest = json.loads(self.original)
        self.model = "adpretko/celerity-271m-4k-ad0p0"

    def test_frozen_manifest_is_read_only(self):
        before = copy.deepcopy(self.manifest)
        spec = validate_manifest(self.manifest, self.model)
        self.assertEqual(spec["revision"], "be032b277561058f3215d0862ba282c164e73d35")
        self.assertEqual(self.manifest, before)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_missing_frozen_indices_and_changed_scoring_rejected(self):
        for change in ("sample_indices", "reference_policy", "common_context", "tasks"):
            doc = copy.deepcopy(self.manifest)
            if change == "sample_indices":
                del doc["sample_indices"]["narrativeqa"]
            elif change == "reference_policy":
                doc["settings"][change] = "best"
            elif change == "common_context":
                doc["settings"][change] = 8192
            else:
                doc["settings"][change] = ["narrativeqa"]
            with self.assertRaises(ValueError):
                validate_manifest(doc, self.model)

    def test_exclusive_output_never_overwrites_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as folder:
            path = Path(folder) / "success.json"
            save_new(path, {"status": "passed"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_new(path, {"status": "failed"})
            self.assertEqual(path.read_bytes(), before)
