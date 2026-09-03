from __future__ import annotations

import json
import unittest
from pathlib import Path

import digital_doctor


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"


class RepositoryLayoutTests(unittest.TestCase):
    def test_runtime_package_uses_digital_doctor_namespace(self) -> None:
        self.assertEqual(digital_doctor.__name__, "digital_doctor")
        self.assertTrue((REPO_ROOT / "digital_doctor" / "core" / "session.py").is_file())

    def test_training_workspace_contains_registered_ocd_datasets(self) -> None:
        registry = json.loads(
            (TRAIN_ROOT / "data" / "dataset_info.json").read_text(encoding="utf-8")
        )
        for dataset_name in ("ocd_train", "ocd_test", "ocd_train_v2", "ocd_test_v2"):
            self.assertIn(dataset_name, registry)
            dataset_path = TRAIN_ROOT / "data" / registry[dataset_name]["file_name"]
            self.assertTrue(dataset_path.is_file(), dataset_path)

    def test_training_migration_excludes_local_state_and_uses_portable_paths(self) -> None:
        self.assertFalse((TRAIN_ROOT / ".git").exists())
        self.assertFalse((TRAIN_ROOT / ".env.local").exists())
        self.assertFalse((TRAIN_ROOT / "saves").exists())

        experiment_scripts = sorted((TRAIN_ROOT / "dev" / "scripts").glob("*/*.sh"))
        self.assertTrue(experiment_scripts)
        for script in experiment_scripts:
            content = script.read_text(encoding="utf-8")
            self.assertNotIn("/workspace", content, script)
            self.assertIn("TRAIN_DIR=", content, script)


if __name__ == "__main__":
    unittest.main()
