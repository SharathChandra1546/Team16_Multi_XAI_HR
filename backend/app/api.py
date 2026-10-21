"""HTTP API (Starlette, the ASGI toolkit FastAPI is built on).

Run:  uvicorn app.api:app --port 8000
Security: HTTPS redirect + HSTS (FORCE_HTTPS=1), security headers, CORS allow-list, per-IP rate
limiting, request size limits, honeypot spam check, admin token from the server environment only.
"""
from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from pathlib import Path

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .config import settings
from .schemas import (
    ChatIntakeRequest, ExplanationRequest, FeedbackRequest, FollowupRequest, IntakeRequest, RefinementRequest,
    WhatIfRequest,
)
from .service import Service, SpamError

MAX_JSON_BYTES = 256_000
MAX_UPLOAD_BYTES = 2_000_000
SPA_ROUTES = {"/", "/explain", "/intake", "/chat-intake", "/batch", "/fairness", "/admin", "/privacy", "/terms"}

_service: Service | None = None


def service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


def err(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


async def parse(request: Request, model: type[BaseModel]):
    raw = await request.body()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Request body is too large")
    try:
        return model.model_validate_json(raw or b"{}")
    except ValidationError as e:
        fields = {".".join(str(p) for p in x["loc"]): x["msg"] for x in e.errors()}
        raise InputError(fields) from None


class InputError(Exception):
    def __init__(self, fields: dict):
        self.fields = fields


def handle(fn):
    async def wrapper(request: Request):
        try:
            return await fn(request)
        except InputError as e:
            return err(422, "Some fields are invalid.", fields=e.fields)
        except SpamError:
            return err(400, "Request rejected.")
        except PermissionError as e:
            return err(403, str(e))
        except KeyError as e:
            return err(404, str(e).strip("'"))
        except FileNotFoundError as e:
            return err(503, str(e))
        except ValueError as e:
            return err(400, str(e))
    return wrapper


def require_admin(request: Request) -> str:
    if not settings.admin_token:
        raise PermissionError("Admin is disabled. Set ADMIN_TOKEN on the server to enable it.")
    token = request.headers.get("x-admin-token", "")
    if not hmac.compare_digest(token.encode(), settings.admin_token.encode()):
        raise PermissionError("Invalid admin token.")
    return "admin"


# ---------------------------------------------------------------- endpoints
@handle
async def health(request: Request):
    return JSONResponse({"ok": True})


@handle
async def config(request: Request):
    return JSONResponse(service().config())


@handle
async def domains(request: Request):
    return JSONResponse(service().domain_cards())


@handle
async def domain_schema(request: Request):
    d = service().registry.get(request.path_params["domain"])
    return JSONResponse(d.schema(request.query_params.get("variant", "main")) | {"sliders": service().slider_specs(d.name)})


@handle
async def examples(request: Request):
    return JSONResponse(service().examples(request.path_params["domain"]))


@handle
async def glossary(request: Request):
    return JSONResponse(service().glossary(request.path_params["domain"]))


@handle
async def aggregate(request: Request):
    return JSONResponse(service().aggregate(request.path_params["domain"]))


@handle
async def fairness(request: Request):
    return JSONResponse(service().fairness(request.path_params["domain"]))


@handle
async def explain(request: Request):
    req = await parse(request, ExplanationRequest)
    return JSONResponse(service().explain(req).model_dump(mode="json"))


@handle
async def refine(request: Request):
    req = await parse(request, RefinementRequest)
    return JSONResponse(service().refine(req).model_dump(mode="json"))


@handle
async def feedback(request: Request):
    req = await parse(request, FeedbackRequest)
    return JSONResponse(service().feedback(req))


@handle
async def whatif(request: Request):
    req = await parse(request, WhatIfRequest)
    return JSONResponse(service().whatif(req))


@handle
async def intake(request: Request):
    req = await parse(request, IntakeRequest)
    return JSONResponse(service().intake(req))


@handle
async def chat_intake(request: Request):
    req = await parse(request, ChatIntakeRequest)
    return JSONResponse(service().chat_intake(req))


@handle
async def pdf_intake(request: Request):
    form = await request.form(max_files=1, max_fields=5)
    if form.get("website"):
        raise SpamError("honeypot")
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise ValueError("Attach a PDF file in the 'file' field")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 2 MB")
    consent = str(form.get("consent_cloud", "")).lower() in ("1", "true", "yes", "on")
    return JSONResponse(service().pdf_intake(request.path_params["domain"], data, consent))


@handle
async def followup(request: Request):
    req = await parse(request, FollowupRequest)
    return JSONResponse(service().followup(req).model_dump(mode="json"))


@handle
async def profile(request: Request):
    return JSONResponse(service().profile_view(request.query_params.get("user_id", "anonymous")))


@handle
async def batch_upload(request: Request):
    form = await request.form(max_files=1, max_fields=5)
    if form.get("website"):
        raise SpamError("honeypot")
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise ValueError("Attach a CSV file in the 'file' field")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 2 MB")
    return JSONResponse(service().batch_upload(request.path_params["domain"], data))


@handle
async def batch_explain(request: Request):
    body = await request.json() if (await request.body()) else {}
    level = body.get("level", "intermediate")
    if level not in ("novice", "intermediate", "expert"):
        raise ValueError("Unknown level")
    resp = service().batch_explain(request.path_params["batch_id"], int(request.path_params["row"]), level,
                                   bool(body.get("consent_cloud", False)), str(body.get("user_id", "anonymous"))[:64])
    return JSONResponse(resp.model_dump(mode="json"))


@handle
async def admin_traces(request: Request):
    require_admin(request)
    return JSONResponse(service().store.traces(int(request.query_params.get("limit", 50))))


@handle
async def admin_trace(request: Request):
    require_admin(request)
    rec = service().store.get_trace(request.path_params["trace_id"])
    if rec is None:
        raise KeyError("Unknown trace id")
    return JSONResponse(rec.model_dump(mode="json"))


@handle
async def admin_audit(request: Request):
    """Archivable audit document for one trace: evidence, plan, attempts, provenance, agent trace (admin only)."""
    require_admin(request)
    return JSONResponse(service().audit(request.path_params["trace_id"]))


@handle
async def admin_prompts(request: Request):
    require_admin(request)
    return JSONResponse(service().store.prompts("generator"))


@handle
async def admin_propose(request: Request):
    require_admin(request)
    return JSONResponse(service().propose_prompt())


@handle
async def admin_approve(request: Request):
    who = require_admin(request)
    return JSONResponse(service().approve_prompt(int(request.path_params["prompt_id"]), who))


@handle
async def admin_summary(request: Request):
    require_admin(request)
    return JSONResponse(service().feedback_summary())


# ---------------------------------------------------------------- middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.hits: dict[str, deque] = defaultdict(deque)

    def client_ip(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd and settings.force_https:  # behind a TLS-terminating proxy
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if settings.force_https:
            proto = request.headers.get("x-forwarded-proto", request.url.scheme)
            if proto == "http":
                return RedirectResponse(str(request.url.replace(scheme="https")), status_code=308)
        if request.method == "POST" and request.url.path.startswith("/api/"):
            ip = self.client_ip(request)
            q = self.hits[ip]
            now = time.monotonic()
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= settings.rate_limit_per_min:
                return err(429, "Too many requests. Please wait a minute and try again.")
            q.append(now)
        response: Response = await call_next(request)
        h = response.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["Referrer-Policy"] = "strict-origin-when-cross-origin"
        h["X-Frame-Options"] = "DENY"
        h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if settings.force_https:
            h["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        if request.url.path.startswith("/api/"):
            h["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            h["Cache-Control"] = "no-store"
        else:
            h["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://plausible.io; connect-src 'self' https://plausible.io; "
                "img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self'; frame-ancestors 'none'; "
                "base-uri 'self'; form-action 'self'" + ("; upgrade-insecure-requests" if settings.force_https else "")
            )
        return response


# ---------------------------------------------------------------- static frontend (optional single-server deploy)
def static_routes() -> list:
    if not settings.static_dir:
        return []
    root = Path(settings.static_dir)
    if not (root / "index.html").exists():
        return []

    async def spa(request: Request):
        path = request.url.path.rstrip("/") or "/"
        if path.startswith("/api/"):
            return err(404, "Not found")
        if path in SPA_ROUTES:
            return FileResponse(root / "index.html")
        f = (root / path.lstrip("/")).resolve()
        if root.resolve() in f.parents and f.is_file():
            return FileResponse(f)
        return FileResponse(root / "404.html", status_code=404)

    return [Mount("/assets", app=StaticFiles(directory=root / "assets"), name="assets"), Route("/{path:path}", spa)]


routes = [
    Route("/api/health", health),
    Route("/api/config", config),
    Route("/api/domains", domains),
    Route("/api/domains/{domain}/schema", domain_schema),
    Route("/api/domains/{domain}/examples", examples),
    Route("/api/domains/{domain}/glossary", glossary),
    Route("/api/domains/{domain}/aggregate", aggregate),
    Route("/api/domains/{domain}/fairness", fairness),
    Route("/api/explain", explain, methods=["POST"]),
    Route("/api/refine", refine, methods=["POST"]),
    Route("/api/feedback", feedback, methods=["POST"]),
    Route("/api/whatif", whatif, methods=["POST"]),
    Route("/api/intake", intake, methods=["POST"]),
    Route("/api/intake/chat", chat_intake, methods=["POST"]),
    Route("/api/domains/{domain}/intake/pdf", pdf_intake, methods=["POST"]),
    Route("/api/followup", followup, methods=["POST"]),
    Route("/api/profile", profile),
    Route("/api/domains/{domain}/batch", batch_upload, methods=["POST"]),
    Route("/api/batch/{batch_id}/explain/{row:int}", batch_explain, methods=["POST"]),
    Route("/api/admin/traces", admin_traces),
    Route("/api/admin/traces/{trace_id}", admin_trace),
    Route("/api/admin/traces/{trace_id}/audit", admin_audit),
    Route("/api/admin/prompts", admin_prompts),
    Route("/api/admin/prompts/propose", admin_propose, methods=["POST"]),
    Route("/api/admin/prompts/{prompt_id:int}/approve", admin_approve, methods=["POST"]),
    Route("/api/admin/summary", admin_summary),
]


def create_app() -> Starlette:
    return Starlette(
        routes=routes + static_routes(),
        middleware=[
            Middleware(SecurityMiddleware),
            Middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Admin-Token"]),
        ],
    )


app = create_app()
