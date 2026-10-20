"""Shared fixtures. Tests run on whatever model is in backend/models (trained on the real CSV or the stand-in)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
_tmp = tempfile.mkdtemp(prefix="mxai-test-")
os.environ["DB_PATH"] = os.path.join(_tmp, "test.sqlite3")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["MOCK_FAULT_RATE"] = "0"

if not (BACKEND / "models" / "hr_main.joblib").exists():  # first run on a fresh checkout
    subprocess.run([sys.executable, str(BACKEND / "scripts" / "train_hr.py"), "--standin", "--fast"], check=True)


@pytest.fixture(scope="session")
def registry():
    from app.domains.registry import default_registry

    return default_registry()


@pytest.fixture(scope="session")
def hr(registry):
    return registry.get("hr")


@pytest.fixture(scope="session")
def instances(hr):
    rt = hr.runtime("main")
    return [{f: float(row[f]) for f in rt.features} for _, row in rt.examples.head(40).iterrows()]


@pytest.fixture(scope="session")
def audit_instances(hr):
    rt = hr.runtime("audit")
    return [{f: float(row[f]) for f in rt.features} for _, row in rt.examples.head(15).iterrows()]


@pytest.fixture(scope="session")
def service(registry):
    from app.service import Service

    return Service(registry=registry)


@pytest.fixture(scope="session")
def package_and_plan(hr, instances):
    from app.agents.planner import make_plan
    from app.xai.evidence import build_package

    pkg = build_package(hr, instances[0], "intermediate", "explain")
    return pkg, make_plan(pkg, "intermediate", "explain")
