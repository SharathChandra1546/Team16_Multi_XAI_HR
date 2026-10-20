"""Check (or obtain) the IBM HR Analytics Employee Attrition dataset.

The dataset is hosted on Kaggle, which requires a login, so this script does not
download it silently. Options:
  1. Download manually from the Kaggle page and save the CSV into backend/data/.
  2. With the Kaggle CLI configured (kaggle.json), run:
       python scripts/fetch_hr.py --kaggle
  3. For software tests only (never for reported results):
       python scripts/fetch_hr.py --standin
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import DATA_DIR  # noqa: E402
from app.domains.hr.data import IBM_FILENAME, IBM_PATH, KAGGLE_URL, STANDIN_PATH, DataError, generate_standin, load_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="download with the Kaggle CLI")
    ap.add_argument("--standin", action="store_true", help="write a synthetic stand-in for software tests")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.standin:
        generate_standin().to_csv(STANDIN_PATH, index=False)
        print(f"Wrote synthetic stand-in to {STANDIN_PATH}. FOR SOFTWARE TESTS ONLY.")
        return 0

    if args.kaggle and not IBM_PATH.exists():
        if not shutil.which("kaggle"):
            print("Kaggle CLI not found. Install with `pip install kaggle` and add your kaggle.json token.")
            return 1
        subprocess.run(["kaggle", "datasets", "download", "-d", "pavansubhasht/ibm-hr-analytics-attrition-dataset",
                        "-p", str(DATA_DIR)], check=True)
        for z in DATA_DIR.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA_DIR)
            z.unlink()

    if not IBM_PATH.exists():
        print(f"Dataset not found.\n  Download: {KAGGLE_URL}\n  Save as:  {IBM_PATH}")
        return 1
    try:
        _, _, info = load_dataset(IBM_PATH)
    except DataError as e:
        print(f"Dataset check failed: {e}")
        return 1
    print(f"OK: {IBM_FILENAME}: {info['rows']} rows, {info['leavers']} leavers ({info['attrition_rate']:.1%}).")
    for w in info["warnings"]:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
