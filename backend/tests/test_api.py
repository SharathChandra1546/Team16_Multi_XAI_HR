"""HTTP API: validation, spam protection, rate limit, admin auth, security headers, HTTPS redirect."""
from starlette.testclient import TestClient


def _client():
    from app import api

    return TestClient(api.create_app())


def _raw(hr, x):
    return {f: hr.readable(f, v) if hr.spec(f).categorical else v for f, v in x.items()}


def test_health_and_headers():
    c = _client()
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]
    for h in ["x-content-type-options", "x-frame-options", "referrer-policy", "content-security-policy"]:
        assert h in r.headers


def test_schema_has_no_secrets():
    c = _client()
    body = c.get("/api/config").text + c.get("/api/domains/hr/schema").text
    for s in ["API_KEY", "sk-", "AIza", "gsk_", "ADMIN_TOKEN"]:
        assert s not in body


def test_explain_and_validation(hr, instances):
    c = _client()
    r = c.post("/api/explain", json={"query": "Why is this employee flagged?", "instance": _raw(hr, instances[2])})
    assert r.status_code == 200 and r.json()["status"] in ("verified", "escalated", "fallback")
    bad = _raw(hr, instances[2]) | {"OverTime": "Maybe", "MonthlyIncome": -5}
    r = c.post("/api/explain", json={"query": "Why?", "instance": bad})
    assert r.status_code == 200 and r.json()["status"] == "error" and "Overtime" in r.json()["message"]
    r = c.post("/api/explain", json={"query": "x" * 2000, "instance": {}})
    assert r.status_code == 422
    r = c.post("/api/feedback", json={"trace_id": "nope", "rating": 9})
    assert r.status_code == 422


def test_honeypot(hr, instances):
    c = _client()
    r = c.post("/api/explain", json={"query": "Why?", "instance": _raw(hr, instances[0]), "website": "x"})
    assert r.status_code == 400


def test_admin_requires_token():
    from app.config import settings

    c = _client()
    old = settings.admin_token
    try:
        settings.admin_token = ""
        assert c.get("/api/admin/traces").status_code == 403
        settings.admin_token = "secret-test-token"
        assert c.get("/api/admin/traces").status_code == 403
        assert c.get("/api/admin/traces", headers={"X-Admin-Token": "secret-test-token"}).status_code == 200
    finally:
        settings.admin_token = old


def test_rate_limit():
    from app.config import settings

    old = settings.rate_limit_per_min
    settings.rate_limit_per_min = 3
    try:
        c = _client()
        codes = [c.post("/api/intake", json={"instance": {}}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
    finally:
        settings.rate_limit_per_min = old


def test_https_redirect_and_hsts():
    from app.config import settings

    settings.force_https = True
    try:
        c = _client()
        r = c.get("/api/health", headers={"x-forwarded-proto": "http"}, follow_redirects=False)
        assert r.status_code == 308 and r.headers["location"].startswith("https://")
        r = c.get("/api/health", headers={"x-forwarded-proto": "https"})
        assert r.status_code == 200 and "strict-transport-security" in r.headers
    finally:
        settings.force_https = False


def test_whatif_rejects_immutable(hr, instances):
    c = _client()
    r = c.post("/api/whatif", json={"instance": _raw(hr, instances[0]), "changes": {"YearsAtCompany": 1}})
    assert r.status_code == 400
    r = c.post("/api/whatif", json={"instance": _raw(hr, instances[0]), "changes": {"OverTime": "No"}})
    assert r.status_code == 200 and "changed" in r.json()


def test_batch_upload(hr):
    import pandas as pd
    from app.domains.hr.data import STANDIN_PATH

    c = _client()
    csv = pd.read_csv(STANDIN_PATH).head(25).to_csv(index=False).encode()
    r = c.post("/api/domains/hr/batch", files={"file": ("x.csv", csv, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_valid"] == 25 and body["rows"][0]["probability"] >= body["rows"][-1]["probability"]
    r = c.post("/api/domains/hr/batch", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400
