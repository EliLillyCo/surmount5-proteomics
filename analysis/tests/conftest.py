from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATHS_FILE = ROOT / "analysis" / "tests" / "paths.test.yaml"

# Collection imports transitively reach paths.py before individual tests can
# monkeypatch the environment, so always pin pytest to the tracked test config.
os.environ["SURMOUNT5_PATHS_YAML"] = str(TEST_PATHS_FILE)
