import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kinelo_data.py"
SPEC = importlib.util.spec_from_file_location("kinelo_data", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class KineloDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        registry = {
            "datasets": {
                "pd4t_derived": {"kind": "local", "path": "clip"},
                "hubu_fis": {"kind": "gdrive", "status": "not_restored"},
            }
        }
        (self.root / "data_registry.json").write_text(json.dumps(registry))

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_dataset_resolves_under_root(self):
        self.assertEqual(
            MODULE.data_path("pd4t_derived", "finger_tapping", root=self.root),
            self.root / "clip" / "finger_tapping",
        )

    def test_root_comes_from_environment(self):
        with mock.patch.dict(os.environ, {"KINELO_DATA_ROOT": str(self.root)}):
            self.assertEqual(MODULE.data_path("pd4t_derived"), self.root / "clip")

    def test_dataset_not_on_drive_fails_loudly(self):
        with self.assertRaises(MODULE.DatasetNotLocalError):
            MODULE.data_path("hubu_fis", root=self.root)

    def test_unknown_dataset_lists_registered_names(self):
        with self.assertRaisesRegex(KeyError, "pd4t_derived"):
            MODULE.data_path("missing", root=self.root)


if __name__ == "__main__":
    unittest.main()
