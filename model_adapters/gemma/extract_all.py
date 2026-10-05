"""Extract Gemma 3 12B activations for the noun-metaphor dataset."""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared.extract_runner import cli  # noqa: E402
import config  # noqa: E402
import model  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli(model, config.get_config, config.SPEC.key))
