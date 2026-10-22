"""Make sure every domain has a trained model, preferring the real dataset.

For each domain (HR, finance, health):
- real dataset present and the current model was not trained on that exact file -> train on it;
- no real dataset and no model -> train on the synthetic stand-in (software test only) and say so;
- otherwise keep the existing model.

    python scripts/ensure_model.py            # all domains
    python scripts/ensure_model.py --fast     # quicker training (fewer CV repeats / bootstraps)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def domains() -> list[dict]:
    from app.domains.finance import data as fin
    from app.domains.health import data as hea
    from app.domains.hr.data import IBM_PATH, KAGGLE_URL

    return [
        {"name": "hr", "real": IBM_PATH, "url": KAGGLE_URL, "script": ["scripts/train_hr.py"],
         "file": "WA_Fn-UseC_-HR-Employee-Attrition.csv"},
        {"name": "finance", "real": fin.UCI_PATH, "url": fin.UCI_URL,
         "script": ["scripts/train_domain.py", "--domain", "finance"], "file": fin.UCI_PATH.name},
        {"name": "health", "real": hea.REAL_PATH, "url": hea.KAGGLE_URL,
         "script": ["scripts/train_domain.py", "--domain", "health"], "file": hea.REAL_PATH.name},
    ]


def trained_on(name: str) -> dict:
    try:
        return json.loads((ROOT / "models" / f"{name}_manifest.json").read_text(encoding="utf-8"))["dataset"]
    except Exception:
        return {}


def run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="quicker training, for trying the app")
    args = ap.parse_args()
    fast = ["--fast"] if args.fast else []
    missing = []
    for d in domains():
        ds = trained_on(d["name"])
        if d["real"].exists():
            digest = hashlib.sha256(d["real"].read_bytes()).hexdigest()[:10]
            if d["name"] == "hr":
                run(["scripts/fetch_hr.py"])
            if ds.get("sha256_prefix") == digest and "standin" not in str(ds.get("source", "")):
                print(f"[{d['name']}] model already trained on {d['file']}.")
                continue
            print(f"[{d['name']}] training on {d['file']} (a few minutes)...")
            run(d["script"] + fast)
            continue
        missing.append(d)
        if ds:
            print(f"[{d['name']}] {d['file']} not found; keeping the existing model (trained on: {ds.get('source')}).")
        else:
            print(f"[{d['name']}] {d['file']} not found; training on the SYNTHETIC STAND-IN (software test only)...")
            run(d["script"] + ["--standin"] + fast)
    if missing:
        print("\nFor real results, put these files in backend\\data\\ and run setup again:")
        for d in missing:
            print(f"  {d['file']:42s} {d['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
