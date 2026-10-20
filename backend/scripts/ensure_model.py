"""Make sure a trained model exists, preferring the real IBM dataset.

- IBM CSV present and the current model was not trained on it -> train on the IBM data.
- No IBM CSV and no model -> train on the synthetic stand-in (software test only) and say so.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.domains.hr.data import IBM_PATH, KAGGLE_URL  # noqa: E402

MANIFEST = ROOT / "models" / "hr_manifest.json"


def current() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["dataset"]
    except Exception:
        return {}


def run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> int:
    ds = current()
    if IBM_PATH.exists():
        digest = hashlib.sha256(IBM_PATH.read_bytes()).hexdigest()[:10]
        run(["scripts/fetch_hr.py"])
        if ds.get("source") != "ibm_kaggle" or ds.get("sha256_prefix") != digest:
            print("Training on the IBM HR dataset (about 2 to 4 minutes)...")
            run(["scripts/train_hr.py"])
        else:
            print("Model already trained on the IBM HR dataset.")
        return 0
    print(f"IBM dataset not found at {IBM_PATH}.\nDownload it from {KAGGLE_URL} for real results.")
    if not ds:
        print("Training on the SYNTHETIC STAND-IN so the app can run (software test only)...")
        run(["scripts/train_hr.py", "--standin"])
    else:
        print(f"Using the existing model (trained on: {ds.get('source')}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
