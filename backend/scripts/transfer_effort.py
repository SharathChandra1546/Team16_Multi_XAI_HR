"""Domain-transfer effort: how much code OUTSIDE a domain's package changed when that domain was added.

  python scripts/transfer_effort.py --domain finance --base <commit before the domain> [--head HEAD]

Counts added/removed lines per file between two git revisions and splits them into:
- inside:  backend/app/domains/<domain>/ (the domain package itself, expected to change)
- registry: backend/app/domains/registry.py (one line to register the domain, expected)
- data/tests/eval/docs for the domain: backend/tests/, backend/eval/, backend/scripts/, docs/ (expected)
- generic: every other file under backend/app/ (should be zero for a truly domain-agnostic framework)

The "generic" count is the paper's quantitative evidence for the domain-agnostic claim. Report it per domain
together with the list of generic files touched and why.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def numstat(base: str, head: str) -> list[tuple[int, int, str]]:
    out = subprocess.run(["git", "-C", str(ROOT), "diff", "--numstat", f"{base}..{head}"], check=True,
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        a, d, path = line.split("\t", 2)
        rows.append((int(a) if a != "-" else 0, int(d) if d != "-" else 0, path))
    return rows


def classify(path: str, domain: str) -> str:
    if path.startswith(f"backend/app/domains/{domain}/"):
        return "inside"
    if path == "backend/app/domains/registry.py":
        return "registry"
    if path.startswith(("backend/tests/", "backend/eval/", "backend/scripts/", "docs/")):
        return "tests_eval_docs"
    if path.startswith("backend/app/"):
        return "generic"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--base", required=True, help="revision before the domain was added")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    groups: dict[str, dict] = {}
    for add, rem, path in numstat(a.base, a.head):
        g = groups.setdefault(classify(path, a.domain), {"added": 0, "removed": 0, "files": []})
        g["added"] += add
        g["removed"] += rem
        g["files"].append(path)
    res = {"domain": a.domain, "base": a.base, "head": a.head, "groups": groups,
           "generic_lines_changed": groups.get("generic", {}).get("added", 0) + groups.get("generic", {}).get("removed", 0)}
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Transfer effort for domain '{a.domain}' ({a.base}..{a.head})")
        for k in ("inside", "registry", "tests_eval_docs", "generic", "other"):
            g = groups.get(k)
            if g:
                print(f"  {k:16s} +{g['added']:<6d} -{g['removed']:<6d} {len(g['files'])} files")
        print(f"\n  Lines changed in generic modules: {res['generic_lines_changed']}")
        for f in groups.get("generic", {}).get("files", []):
            print(f"    {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
