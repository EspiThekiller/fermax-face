"""FastAPI Microservice for Fermax Face Recognition with Enterprise-Grade Security.

Features:
- Secure multi-factor authentication (Web sessions + API tokens).
- First-boot random 10-char admin password with single-time log display.
- Mandatory elimination/replacement of 'admin' on first login.
- Named API tokens stored via SHA-256 hashes (never retrievable in plaintext).
- Protection against brute-force login attempts (rate limiting).
- 3-day capture history with automatic purging (>72h retention).
- Self-training panel to assign captured camera snapshots to registered people.
- Real-time CPU face recognition powered by InsightFace (buffalo_s).
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import FaceRecognitionEngine
from .history import HistoryManager
from .security import SecurityManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOGGER = logging.getLogger("fermax_face_api")

KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", "/app/known_faces")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
DET_SIZE = int(os.getenv("DET_SIZE", "320"))

# Initialize Security, Recognition Engine and 3-Day History Manager
security = SecurityManager(data_dir=DATA_DIR)
history = HistoryManager(data_dir=DATA_DIR)
engine = FaceRecognitionEngine(
    known_faces_dir=KNOWN_FACES_DIR,
    model_name="buffalo_s",
    det_size=DET_SIZE,
    similarity_threshold=SIMILARITY_THRESHOLD,
)


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class SetupAdminRequest(BaseModel):
    new_username: str = Field(..., min_length=3)
    new_password: str = Field(..., min_length=6)


class CreateTokenRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class AssignHistoryRequest(BaseModel):
    capture_id: str = Field(..., min_length=1)
    person_name: str = Field(..., min_length=1)


# -----------------------------------------------------------------------------
# Authentication Dependencies
# -----------------------------------------------------------------------------
def get_api_auth(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
    session_id: str | None = Cookie(None),
) -> str:
    """Validate incoming request for API access.
    
    Accepts:
    1. 'X-API-Key' header (API Token, e.g. from Home Assistant).
    2. 'Authorization: Bearer <token>' header.
    3. Active web session cookie ('session_id').
    """
    # 1. Check API token via X-API-Key header
    if x_api_key and security.verify_api_token(x_api_key):
        return "api_token"

    # 2. Check API token via Authorization: Bearer header
    if authorization and authorization.startswith("Bearer "):
        bearer_token = authorization[7:].strip()
        if security.verify_api_token(bearer_token):
            return "api_token"

    # 3. Check active web session
    if session_id:
        is_valid, username, _ = security.verify_session(session_id)
        if is_valid and username:
            return username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Acceso no autorizado. Proporciona un Token de API válido ('X-API-Key') o inicia sesión en el panel.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_admin_session(
    request: Request,
    session_id: str | None = Cookie(None),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Validate active web administrator session."""
    active_sid = session_id or x_session_id
    if not active_sid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión no iniciada. Por favor identifícate en el panel.",
        )

    is_valid, username, must_change = security.verify_session(active_sid)
    if not is_valid or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La sesión ha expirado o no es válida.",
        )

    return {"username": username, "must_change": must_change, "session_id": active_sid}


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown."""
    _LOGGER.info("Starting Fermax Face Recognition Microservice...")
    engine.initialize()
    history.cleanup_expired()
    yield
    _LOGGER.info("Shutting down Fermax Face Recognition Microservice.")


app = FastAPI(
    title="Fermax Face Recognition Service",
    description="Microservicio CPU ultraligero y seguro para apertura facial en Fermax NOWIFI",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Health & Probes
# -----------------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silent 204 response for browser favicon requests."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Healthcheck endpoint (public)."""
    return {
        "status": "ok",
        "engine": "InsightFace (buffalo_s)",
        "device": "CPU",
        "registered_people": len(engine.known_db),
        "history_captures_count": len(history.list_captures()),
        "security_initialized": security.data.get("first_boot_completed", False),
    }


# -----------------------------------------------------------------------------
# Authentication Endpoints (Web Panel)
# -----------------------------------------------------------------------------
@app.get("/api/v1/auth/me")
async def auth_status(
    session_id: str | None = Cookie(None),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Check current authentication status."""
    active_sid = session_id or x_session_id
    if not active_sid:
        return {"authenticated": False, "user": None, "must_change": False}

    is_valid, username, must_change = security.verify_session(active_sid)
    return {
        "authenticated": is_valid,
        "user": username,
        "must_change": must_change,
    }


@app.post("/api/v1/auth/login")
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    """Authenticate user with rate-limiting brute force protection."""
    client_ip = request.client.host if request.client else "unknown"

    if security.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiados intentos fallidos. Por seguridad, la IP ha sido bloqueada temporalmente durante 60 segundos.",
        )

    success, user, must_change = security.authenticate_user(
        payload.username.strip(), payload.password.strip(), client_ip=client_ip
    )

    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
        )

    session_id = security.create_session(user)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )

    return {
        "status": "ok",
        "user": user,
        "must_change": must_change,
        "session_id": session_id,
    }


@app.post("/api/v1/auth/setup-admin")
async def setup_admin(
    payload: SetupAdminRequest,
    response: Response,
    admin_session: dict[str, Any] = Depends(get_admin_session),
) -> dict[str, Any]:
    """Mandatory initial configuration: delete temporary 'admin' user and set new credentials."""
    if not admin_session.get("must_change"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El usuario administrador ya ha sido configurado.",
        )

    try:
        security.setup_new_admin(payload.new_username, payload.new_password)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    new_sid = security.create_session(payload.new_username)
    response.set_cookie(
        key="session_id",
        value=new_sid,
        httponly=True,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )

    return {
        "status": "ok",
        "message": f"Usuario inicial 'admin' eliminado permanentemente. Administrador '{payload.new_username}' activado.",
        "user": payload.new_username,
        "session_id": new_sid,
    }


@app.post("/api/v1/auth/logout")
async def logout(
    response: Response,
    session_id: str | None = Cookie(None),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Terminate current web session and clear cookie."""
    active_sid = session_id or x_session_id
    security.delete_session(active_sid)
    response.delete_cookie(key="session_id")
    return {"status": "ok", "message": "Sesión cerrada correctamente."}


# -----------------------------------------------------------------------------
# API Tokens Management Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/v1/tokens")
async def list_tokens(
    admin_session: dict[str, Any] = Depends(get_admin_session),
) -> list[dict[str, Any]]:
    """List active API tokens with metadata. Plaintext tokens cannot be viewed."""
    return security.list_api_tokens()


@app.post("/api/v1/tokens")
async def create_token(
    payload: CreateTokenRequest,
    admin_session: dict[str, Any] = Depends(get_admin_session),
) -> dict[str, Any]:
    """Create a new named API token. The plaintext token is returned ONLY ONCE."""
    metadata, raw_token = security.create_api_token(payload.name)
    return {
        "token": raw_token,
        "metadata": metadata,
        "warning": "Copia este token ahora. Por motivos de seguridad, nunca se volverá a mostrar.",
    }


@app.delete("/api/v1/tokens/{token_id}")
async def delete_token(
    token_id: str,
    admin_session: dict[str, Any] = Depends(get_admin_session),
) -> dict[str, Any]:
    """Revoke and permanently delete an API token."""
    deleted = security.delete_api_token(token_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token no encontrado")
    return {"status": "ok", "deleted": token_id}


# -----------------------------------------------------------------------------
# Face Recognition & Database Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/v1/recognize")
async def recognize(
    file: UploadFile = File(...),
    threshold: float | None = Query(None, description="Umbral de similitud (ej: 0.65)"),
    auth: str = Depends(get_api_auth),
) -> dict[str, Any]:
    """Identify person in the uploaded image from Fermax doorbell."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Archivo de imagen vacío")
        result = engine.recognize(content, threshold=threshold)

        # Log visual claro del resultado del reconocimiento para seguimiento en tiempo real
        if result.get("match"):
            _LOGGER.info(
                "✅ AUTORIZADO: '%s' (similitud: %.1f%%, tiempo: %.1f ms) -> Apertura enviada",
                result.get("name"),
                result.get("confidence", 0.0) * 100,
                result.get("execution_time_ms", 0.0),
            )
        elif result.get("detected"):
            _LOGGER.info(
                "❌ NO AUTORIZADO: Rostro detectado pero no coincide con nadie (candidato: '%s', similitud: %.1f%%, umbral: %.1f%%)",
                result.get("candidate_name", "Desconocido"),
                result.get("confidence", 0.0) * 100,
                result.get("threshold_used", 0.65) * 100,
            )
        else:
            _LOGGER.info(
                "⚪ SIN ROSTRO: La imagen analizada no contiene ninguna cara visible (tiempo: %.1f ms)",
                result.get("execution_time_ms", 0.0),
            )

        # Store in 3-day history if face was detected (authorized or unauthorized)
        if result.get("detected"):
            history.record_capture(content, result)

        return result
    except Exception as exc:
        _LOGGER.error("Error recognizing image: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/faces")
async def list_faces(auth: str = Depends(get_api_auth)) -> list[dict[str, Any]]:
    """List all registered authorized people."""
    return engine.list_people()


@app.post("/api/v1/faces/{name}")
async def add_faces(
    name: str,
    files: list[UploadFile] = File(...),
    auth: str = Depends(get_api_auth),
) -> dict[str, Any]:
    """Add one or more reference photos for a person."""
    if not files:
        raise HTTPException(status_code=400, detail="No se seleccionó ninguna foto")

    photos_to_process: list[tuple[bytes, str]] = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        ext = os.path.splitext(f.filename or "photo.jpg")[1] or ".jpg"
        unique_name = f"{uuid.uuid4().hex[:8]}{ext}"
        photos_to_process.append((content, unique_name))

    if not photos_to_process:
        raise HTTPException(status_code=400, detail="Los archivos de imagen estaban vacíos")

    result = engine.register_faces_bulk(name=name, photos=photos_to_process)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))
    return result


@app.delete("/api/v1/faces/{name}")
async def delete_face(
    name: str,
    auth: str = Depends(get_api_auth),
) -> dict[str, Any]:
    """Delete an authorized person and all their photos."""
    deleted = engine.delete_person(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    return {"status": "ok", "deleted": name}


@app.post("/api/v1/reload")
async def reload_faces(auth: str = Depends(get_api_auth)) -> dict[str, Any]:
    """Reload all faces from disk."""
    engine.load_known_faces()
    return {"status": "ok", "people_count": len(engine.known_db)}


# -----------------------------------------------------------------------------
# 3-Day History & Self-Training Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/v1/history")
async def get_history(auth: str = Depends(get_api_auth)) -> list[dict[str, Any]]:
    """List all verified camera captures from the past 3 days (ordered newest first)."""
    return history.list_captures()


@app.get("/api/v1/history/image/{filename}")
async def get_history_image(filename: str, auth: str = Depends(get_api_auth)):
    """Serve capture image for web display."""
    path = history.get_image_path(filename)
    if not path:
        raise HTTPException(status_code=404, detail="Imagen no encontrada en el histórico")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/v1/history/assign")
async def assign_history_capture(
    payload: AssignHistoryRequest,
    auth: str = Depends(get_api_auth),
) -> dict[str, Any]:
    """Assign a snapshot from the 3-day history to a person for self-training."""
    res = history.assign_to_person(
        capture_id=payload.capture_id,
        person_name=payload.person_name,
        engine=engine,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.delete("/api/v1/history/{capture_id}")
async def delete_history_capture(
    capture_id: str,
    auth: str = Depends(get_api_auth),
) -> dict[str, Any]:
    """Delete a single capture from the 3-day history."""
    deleted = history.delete_capture(capture_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Captura no encontrada")
    return {"status": "ok", "deleted": capture_id}


@app.delete("/api/v1/history")
async def clear_all_history(auth: str = Depends(get_api_auth)) -> dict[str, Any]:
    """Purge entire capture history."""
    count = history.clear_all()
    return {"status": "ok", "purged": count}


# -----------------------------------------------------------------------------
# Single Page Web Application (SPA)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Modern, secure single-page web interface with tabs, self-training and security."""
    return """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Fermax Face Recognition & Security</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0b1120;
      --card: #1e293b;
      --card-alt: #162032;
      --border: #334155;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 1.5rem 1rem; display: flex; justify-content: center; min-height: 100vh; }
    .container { max-width: 1040px; width: 100%; }
    
    header { margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; flex-wrap: wrap; gap: 1rem; }
    h1 { font-size: 1.5rem; color: var(--primary); display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: #0369a1; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; text-transform: uppercase; font-weight: bold; }
    .badge-user { background: #1e3a8a; color: #93c5fd; padding: 0.3rem 0.7rem; border-radius: 9999px; font-size: 0.8rem; font-weight: bold; border: 1px solid #3b82f6; }
    
    /* Tabs Navigation */
    .tab-bar { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
    .tab-btn { background: transparent; border: 1px solid transparent; color: var(--text-muted); font-size: 0.95rem; font-weight: 600; padding: 0.6rem 1.2rem; border-radius: 0.5rem; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; gap: 0.5rem; }
    .tab-btn:hover { color: var(--text); background: rgba(56, 189, 248, 0.08); }
    .tab-btn.active { background: var(--card); border-color: var(--border); color: var(--primary); border-bottom-color: var(--card); }
    .tab-badge { background: #0284c7; color: #fff; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 9999px; font-weight: bold; }

    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
    .card-full { grid-column: 1 / -1; }
    h2 { font-size: 1.15rem; margin-bottom: 1rem; color: #e2e8f0; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
    
    label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 500; }
    input[type="text"], input[type="password"], select { width: 100%; padding: 0.65rem 0.8rem; background: #0f172a; border: 1px solid var(--border); border-radius: 0.5rem; color: #fff; margin-bottom: 1rem; font-size: 0.9rem; }
    input:focus, select:focus { outline: none; border-color: var(--primary); }
    
    button { background: var(--primary); color: #0b1120; border: none; font-weight: 600; padding: 0.6rem 1.2rem; border-radius: 0.5rem; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; justify-content: center; gap: 0.4rem; }
    button:hover { background: var(--primary-hover); color: #fff; }
    button.secondary { background: #334155; color: #f1f5f9; padding: 0.4rem 0.8rem; font-size: 0.82rem; }
    button.secondary:hover { background: #475569; }
    button.danger { background: #7f1d1d; color: #fca5a5; padding: 0.4rem 0.75rem; font-size: 0.82rem; }
    button.danger:hover { background: var(--danger); color: #fff; }
    button.logout { background: transparent; border: 1px solid #475569; color: var(--text-muted); font-size: 0.8rem; padding: 0.35rem 0.8rem; }
    button.logout:hover { border-color: var(--danger); color: var(--danger); }
    
    .drop-zone { width: 100%; border: 2px dashed var(--border); border-radius: 0.5rem; min-height: 110px; display: flex; align-items: center; justify-content: center; flex-direction: column; cursor: pointer; padding: 1rem; text-align: center; color: var(--text-muted); margin-bottom: 1rem; transition: 0.2s; }
    .drop-zone.dragover { border-color: var(--primary); background: rgba(56, 189, 248, 0.05); color: var(--primary); }
    .file-count { font-size: 0.85rem; color: var(--accent); margin-top: 0.4rem; font-weight: bold; }
    
    .person-item { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: var(--card-alt); border: 1px solid var(--border); border-radius: 0.5rem; margin-bottom: 0.5rem; gap: 0.5rem; }
    .person-actions { display: flex; gap: 0.4rem; align-items: center; }
    
    .token-table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.85rem; }
    .token-table th, .token-table td { padding: 0.65rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
    .token-table th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; }
    .token-table td { color: #e2e8f0; }
    .token-code { font-family: monospace; background: #0f172a; padding: 0.2rem 0.45rem; border-radius: 0.25rem; color: var(--primary); font-size: 0.85rem; }
    
    /* History Cards */
    .history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .history-card { background: var(--card-alt); border: 1px solid var(--border); border-radius: 0.65rem; overflow: hidden; display: flex; flex-direction: column; transition: transform 0.15s; }
    .history-card:hover { transform: translateY(-2px); border-color: var(--primary); }
    .history-img-wrap { position: relative; width: 100%; height: 180px; background: #000; overflow: hidden; }
    .history-img { width: 100%; height: 100%; object-fit: cover; }
    .history-badge { position: absolute; top: 8px; left: 8px; padding: 0.25rem 0.55rem; border-radius: 0.35rem; font-size: 0.72rem; font-weight: bold; }
    .badge-match { background: rgba(34, 197, 94, 0.95); color: #052e16; }
    .badge-no-match { background: rgba(239, 68, 68, 0.95); color: #450a0a; }
    .history-body { padding: 0.85rem; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
    
    .result-card { margin-top: 1rem; padding: 1rem; border-radius: 0.5rem; display: none; }
    .result-success { background: rgba(34, 197, 94, 0.1); border: 1px solid var(--accent); }
    .result-fail { background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); }
    
    /* Modals & Overlays */
    .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; z-index: 1000; padding: 1rem; }
    .modal-box { background: var(--card); border: 1px solid var(--border); border-radius: 0.85rem; max-width: 480px; width: 100%; padding: 2rem; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5); }
    .alert-banner { background: rgba(245, 158, 11, 0.12); border: 1px solid var(--warning); border-radius: 0.5rem; padding: 0.85rem 1rem; margin-bottom: 1.25rem; font-size: 0.85rem; line-height: 1.4; color: #fde68a; }
    .alert-danger { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger); color: #fca5a5; }
    .secret-box { background: #020617; border: 1px dashed var(--primary); padding: 1rem; border-radius: 0.5rem; font-family: monospace; font-size: 0.95rem; word-break: break-all; color: #38bdf8; margin: 1rem 0; user-select: all; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <!-- Main Header -->
    <header>
      <div>
        <h1>🚪 Fermax Face Recognition</h1>
        <span style="font-size:0.8rem; color:var(--text-muted);">Microservicio de Inteligencia Artificial para apertura de puerta</span>
      </div>
      <div id="headerAuthArea" style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap;">
        <span class="badge">CPU • Buffalo_s (Optimizado)</span>
        <span id="userBadge" class="badge-user" style="display:none;"></span>
        <button id="btnLogout" class="logout" style="display:none;" onclick="logout()">Cerrar Sesión</button>
      </div>
    </header>

    <!-- State 1: Login Form -->
    <div id="loginView" style="display:none; max-width:400px; margin: 3rem auto 0 auto;">
      <div class="card">
        <h2 style="justify-content:center; margin-bottom:1.25rem;">🔐 Iniciar Sesión</h2>
        <div id="loginAlert" class="alert-banner alert-danger" style="display:none;"></div>
        
        <label>Usuario:</label>
        <input type="text" id="loginUsername" placeholder="admin o tu usuario" autocomplete="username">
        
        <label>Contraseña:</label>
        <input type="password" id="loginPassword" placeholder="Contraseña de acceso" autocomplete="current-password" onkeydown="if(event.key==='Enter') doLogin()">
        
        <button onclick="doLogin()" style="width:100%; margin-top:0.5rem;">Entrar al Panel</button>
        <p style="font-size:0.75rem; color:var(--text-muted); text-align:center; margin-top:1rem;">
          ℹ️ Las credenciales se conservan en tu volumen seguro <code>data/security.json</code>.
        </p>
      </div>
    </div>

    <!-- State 2: Mandatory First-Boot Setup Modal -->
    <div id="setupAdminOverlay" class="modal-overlay" style="display:none;">
      <div class="modal-box">
        <h2 style="color:var(--warning); margin-bottom:0.75rem;">⚠️ Configuración Obligatoria</h2>
        <div class="alert-banner">
          Has iniciado sesión con el usuario temporal <strong>admin</strong>. Por motivos de seguridad, es <strong>obligatorio eliminarlo</strong> y configurar tu nuevo usuario y contraseña de administrador definitivos.
        </div>
        <div id="setupAlert" class="alert-banner alert-danger" style="display:none;"></div>

        <label>Nuevo Nombre de Usuario:</label>
        <input type="text" id="setupNewUser" placeholder="Ej: nuevo_admin">

        <label>Nueva Contraseña (mínimo 6 caracteres):</label>
        <input type="password" id="setupNewPass" placeholder="Nueva contraseña segura">

        <label>Confirmar Nueva Contraseña:</label>
        <input type="password" id="setupNewPass2" placeholder="Repite la contraseña">

        <button onclick="submitSetupAdmin()" style="width:100%; background:var(--warning); color:#000;">
          🗑️ Eliminar 'admin' y Guardar Administrador
        </button>
      </div>
    </div>

    <!-- State 3: Main Authenticated Dashboard -->
    <div id="dashboardView" style="display:none;">
      <!-- Tab Bar -->
      <div class="tab-bar">
        <button id="tabBtnMain" class="tab-btn active" onclick="switchTab('main')">
          👥 Personas y Reconocimiento
        </button>
        <button id="tabBtnHistory" class="tab-btn" onclick="switchTab('history')">
          🕒 Historial y Autoentrenamiento (3 días)
          <span id="historyCountBadge" class="tab-badge">0</span>
        </button>
      </div>

      <!-- Tab View 1: Main (Personas, Prueba, Tokens) -->
      <div id="tabViewMain">
        <div class="grid">
          <!-- Card 1: Personas Autorizadas -->
          <div class="card">
            <h2>
              Personas Autorizadas
              <button class="secondary" onclick="loadPeople()">🔄 Refrescar</button>
            </h2>
            <div id="peopleList" style="margin-top:0.75rem; max-height:260px; overflow-y:auto;">
              <p style="color:var(--text-muted); font-size:0.85rem;">Cargando personas...</p>
            </div>

            <h3 style="font-size:0.95rem; margin-top:1.5rem; margin-bottom:0.75rem; border-top:1px solid var(--border); padding-top:1rem;">
              ➕ Registrar Nueva Persona
            </h3>
            <label>Nombre de la Persona:</label>
            <input type="text" id="newName" placeholder="Nombre">

            <label>Fotos de la Cara (subida múltiple):</label>
            <div class="drop-zone" id="dropZoneNew" onclick="document.getElementById('newPhotos').click()">
              <span style="font-size:1.5rem;">📁</span>
              <span>Haz clic o arrastra <strong>una o varias fotos</strong> aquí</span>
              <div id="newPhotoCount" class="file-count"></div>
            </div>
            <input type="file" id="newPhotos" accept="image/*" multiple style="display:none;" onchange="updateNewPhotoCount()">

            <button id="btnRegister" onclick="registerPerson()" style="width:100%;">Registrar Persona y Subir Fotos</button>
          </div>

          <!-- Card 2: Probar Reconocimiento -->
          <div class="card">
            <h2>📸 Probar Reconocimiento</h2>
            <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:1rem;">
              Sube una foto o fotograma para simular la llegada de una persona frente al telefonillo.
            </p>

            <div class="drop-zone" id="dropZoneTest" onclick="document.getElementById('testPhoto').click()">
              <span style="font-size:1.5rem;">🔍</span>
              <span>Haz clic o arrastra una foto para probar</span>
            </div>
            <input type="file" id="testPhoto" accept="image/*" style="display:none;" onchange="runTest()">

            <div id="testResult" class="result-card">
              <h3 id="resultTitle" style="margin-bottom:0.4rem;"></h3>
              <p id="resultDetails" style="font-size:0.85rem; line-height:1.4;"></p>
            </div>
          </div>

          <!-- Card 3: API Tokens Management -->
          <div class="card card-full">
            <h2>
              🔑 Tokens de API (Home Assistant / Integraciones)
              <button onclick="promptCreateToken()">➕ Generar Nuevo Token</button>
            </h2>
            <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:1rem;">
              Genera tokens seguros para autorizar la integración Fermax NOWIFI de Home Assistant sin compartir contraseñas. Los tokens se almacenan protegidos con SHA-256.
            </p>

            <div style="overflow-x:auto;">
              <table class="token-table">
                <thead>
                  <tr>
                    <th>Nombre / Descripción</th>
                    <th>Identificador (Prefijo)</th>
                    <th>Fecha de Creación</th>
                    <th>Último Uso</th>
                    <th style="text-align:right;">Acción</th>
                  </tr>
                </thead>
                <tbody id="tokensTableBody">
                  <tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Cargando tokens...</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab View 2: History & Self-Training (3 days) -->
      <div id="tabViewHistory" style="display:none;">
        <div class="card card-full">
          <h2>
            <span>🕒 Capturas de la Cámara (Últimos 3 Días)</span>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
              <button class="secondary" onclick="loadHistory()">🔄 Refrescar Capturas</button>
              <button class="danger" onclick="clearAllHistory()">🧹 Vaciar Histórico</button>
            </div>
          </h2>
          <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:1rem; line-height:1.4;">
            Cada vez que el telefonillo o Home Assistant envía una foto a la API, queda registrada aquí durante <strong>3 días</strong>. Las fotos con más de 3 días se purgan automáticamente.<br>
            Usa el selector para asignar cualquier foto a una persona autorizada y <strong>autoentrenar el modelo</strong> con ese nuevo ángulo, iluminación o expresión facial.
          </p>

          <div id="historyContainer">
            <p style="color:var(--text-muted); font-size:0.85rem;">Cargando capturas...</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Token Creation One-Time Display Modal -->
  <div id="tokenModal" class="modal-overlay" style="display:none;">
    <div class="modal-box">
      <h2>🎉 Nuevo Token de API Generado</h2>
      <div class="alert-banner">
        ⚠️ <strong>IMPORTANTE:</strong> Por motivos de seguridad, este token <strong>NUNCA se volverá a mostrar</strong>. Cópialo ahora y pégalo en las opciones de la integración en Home Assistant.
      </div>
      <label>Tu Token de Acceso Seguro:</label>
      <div id="rawTokenDisplay" class="secret-box"></div>
      
      <div style="display:flex; gap:0.5rem; margin-top:1.25rem;">
        <button onclick="copyGeneratedToken()" style="flex:1;">📋 Copiar Token</button>
        <button class="secondary" onclick="closeTokenModal()">Cerrar</button>
      </div>
    </div>
  </div>

  <!-- Input oculto para añadir fotos adicionales a una persona existente -->
  <input type="file" id="addMorePhotosInput" accept="image/*" multiple style="display:none;" onchange="submitAdditionalPhotos()">

  <script>
    let currentTargetPerson = null;
    let lastGeneratedSecret = '';
    let registeredPeopleList = [];

    // Switch Tabs
    function switchTab(tab) {
      const btnMain = document.getElementById('tabBtnMain');
      const btnHistory = document.getElementById('tabBtnHistory');
      const viewMain = document.getElementById('tabViewMain');
      const viewHistory = document.getElementById('tabViewHistory');

      if (tab === 'history') {
        btnMain.classList.remove('active');
        btnHistory.classList.add('active');
        viewMain.style.display = 'none';
        viewHistory.style.display = 'block';
        loadHistory();
      } else {
        btnHistory.classList.remove('active');
        btnMain.classList.add('active');
        viewHistory.style.display = 'none';
        viewMain.style.display = 'block';
        loadPeople();
      }
    }

    // Check Auth Status on Load
    async function checkAuth() {
      try {
        const res = await fetch('/api/v1/auth/me');
        const data = await res.json();
        
        if (!data.authenticated) {
          showLoginView();
          return;
        }

        document.getElementById('userBadge').innerText = `👤 ${data.user}`;
        document.getElementById('userBadge').style.display = 'inline-block';
        document.getElementById('btnLogout').style.display = 'inline-flex';

        if (data.must_change) {
          document.getElementById('setupAdminOverlay').style.display = 'flex';
        } else {
          document.getElementById('setupAdminOverlay').style.display = 'none';
          showDashboardView();
        }
      } catch (err) {
        showLoginView();
      }
    }

    function showLoginView() {
      document.getElementById('loginView').style.display = 'block';
      document.getElementById('dashboardView').style.display = 'none';
      document.getElementById('setupAdminOverlay').style.display = 'none';
      document.getElementById('userBadge').style.display = 'none';
      document.getElementById('btnLogout').style.display = 'none';
    }

    function showDashboardView() {
      document.getElementById('loginView').style.display = 'none';
      document.getElementById('dashboardView').style.display = 'block';
      loadPeople();
      loadTokens();
      loadHistoryCount();
    }

    async function doLogin() {
      const user = document.getElementById('loginUsername').value.trim();
      const pass = document.getElementById('loginPassword').value.trim();
      const alertBox = document.getElementById('loginAlert');
      alertBox.style.display = 'none';

      if (!user || !pass) {
        alertBox.innerText = 'Por favor completa usuario y contraseña.';
        alertBox.style.display = 'block';
        return;
      }

      try {
        const res = await fetch('/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user, password: pass })
        });
        const data = await res.json();
        if (res.ok) {
          document.getElementById('loginPassword').value = '';
          checkAuth();
        } else {
          alertBox.innerText = data.detail || 'Error al iniciar sesión.';
          alertBox.style.display = 'block';
        }
      } catch (err) {
        alertBox.innerText = 'Error de conexión con el servicio.';
        alertBox.style.display = 'block';
      }
    }

    async function submitSetupAdmin() {
      const newUser = document.getElementById('setupNewUser').value.trim();
      const pass1 = document.getElementById('setupNewPass').value.trim();
      const pass2 = document.getElementById('setupNewPass2').value.trim();
      const alertBox = document.getElementById('setupAlert');
      alertBox.style.display = 'none';

      if (!newUser || !pass1) {
        alertBox.innerText = 'Por favor completa todos los campos.';
        alertBox.style.display = 'block';
        return;
      }
      if (pass1 !== pass2) {
        alertBox.innerText = 'Las contraseñas no coinciden.';
        alertBox.style.display = 'block';
        return;
      }
      if (pass1.length < 6) {
        alertBox.innerText = 'La contraseña debe tener al menos 6 caracteres.';
        alertBox.style.display = 'block';
        return;
      }

      try {
        const res = await fetch('/api/v1/auth/setup-admin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_username: newUser, new_password: pass1 })
        });
        const data = await res.json();
        if (res.ok) {
          alert('¡Cuenta configurada con éxito! El usuario temporal "admin" ha sido eliminado.');
          document.getElementById('setupAdminOverlay').style.display = 'none';
          checkAuth();
        } else {
          alertBox.innerText = data.detail || 'Error configurando administrador.';
          alertBox.style.display = 'block';
        }
      } catch (err) {
        alertBox.innerText = 'Error de conexión: ' + err;
        alertBox.style.display = 'block';
      }
    }

    async function logout() {
      await fetch('/api/v1/auth/logout', { method: 'POST' });
      showLoginView();
    }

    // Drag and Drop Helper
    function setupDragAndDrop(zoneId, inputId, callback) {
      const zone = document.getElementById(zoneId);
      const input = document.getElementById(inputId);
      ['dragenter', 'dragover'].forEach(name => {
        zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.add('dragover'); });
      });
      ['dragleave', 'drop'].forEach(name => {
        zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.remove('dragover'); });
      });
      zone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
          input.files = e.dataTransfer.files;
          if (callback) callback();
        }
      });
    }

    function updateNewPhotoCount() {
      const files = document.getElementById('newPhotos').files;
      const countDiv = document.getElementById('newPhotoCount');
      if (files.length === 1) {
        countDiv.innerText = `1 foto seleccionada (${files[0].name})`;
      } else if (files.length > 1) {
        countDiv.innerText = `✓ ${files.length} fotos seleccionadas preparadas para subir`;
      } else {
        countDiv.innerText = '';
      }
    }

    // Load Authorized Faces
    async function loadPeople() {
      try {
        const res = await fetch('/api/v1/faces');
        if (res.status === 401) { showLoginView(); return; }
        const list = await res.json();
        registeredPeopleList = list;
        const div = document.getElementById('peopleList');
        if (list.length === 0) {
          div.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">No hay personas registradas aún. Añade una abajo con varias fotos.</p>';
          return;
        }
        div.innerHTML = list.map(p => `
          <div class="person-item">
            <div>
              <strong>${p.name}</strong>
              <div style="font-size:0.75rem; color:var(--text-muted);">${p.photos_count} fotos biométricas</div>
            </div>
            <div class="person-actions">
              <button class="secondary" title="Añadir más fotos" onclick="promptAddPhotos('${p.name}')">➕ Añadir</button>
              <button class="danger" title="Eliminar persona" onclick="deletePerson('${p.name}')">🗑️</button>
            </div>
          </div>
        `).join('');
      } catch (err) {
        document.getElementById('peopleList').innerHTML = `<p style="color:var(--danger); font-size:0.85rem;">Error: ${err}</p>`;
      }
    }

    async function registerPerson() {
      const name = document.getElementById('newName').value.trim();
      const files = document.getElementById('newPhotos').files;
      if (!name) { alert('Introduce un nombre para la persona.'); return; }
      if (!files || files.length === 0) { alert('Selecciona al menos una foto.'); return; }

      const btn = document.getElementById('btnRegister');
      btn.disabled = true;
      btn.innerText = `Procesando ${files.length} fotos...`;

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) { formData.append('files', files[i]); }

      try {
        const res = await fetch(`/api/v1/faces/${encodeURIComponent(name)}`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          alert(`¡${name} registrado con éxito! Fotos biométricas añadidas: ${data.added_photos || files.length}`);
          document.getElementById('newName').value = '';
          document.getElementById('newPhotos').value = '';
          document.getElementById('newPhotoCount').innerText = '';
          loadPeople();
        } else {
          alert('Error: ' + (data.detail || data.message || 'Error desconocido'));
        }
      } catch (err) {
        alert('Error de conexión: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerText = 'Registrar Persona y Subir Fotos';
      }
    }

    function promptAddPhotos(name) {
      currentTargetPerson = name;
      const input = document.getElementById('addMorePhotosInput');
      input.value = '';
      input.click();
    }

    async function submitAdditionalPhotos() {
      const input = document.getElementById('addMorePhotosInput');
      const files = input.files;
      if (!currentTargetPerson || !files || files.length === 0) return;

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) { formData.append('files', files[i]); }

      try {
        const res = await fetch(`/api/v1/faces/${encodeURIComponent(currentTargetPerson)}`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          alert(`¡Fotos añadidas a ${currentTargetPerson}! Nuevas válidas: ${data.added_photos || 0}. Total: ${data.total_photos || 0}`);
          loadPeople();
        } else {
          alert('Error: ' + (data.detail || 'Error procesando fotos'));
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    async function deletePerson(name) {
      if (!confirm(`¿Eliminar permanentemente a ${name} y todas sus fotos?`)) return;
      await fetch(`/api/v1/faces/${encodeURIComponent(name)}`, { method: 'DELETE' });
      loadPeople();
    }

    // Test Recognition
    async function runTest() {
      const fileInput = document.getElementById('testPhoto');
      if (!fileInput.files[0]) return;
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const resDiv = document.getElementById('testResult');
      resDiv.style.display = 'block';
      resDiv.className = 'result-card';
      document.getElementById('resultTitle').innerText = 'Analizando fotograma con IA...';
      document.getElementById('resultDetails').innerText = '';

      try {
        const res = await fetch('/api/v1/recognize', { method: 'POST', body: formData });
        const data = await res.json();
        loadHistoryCount();
        
        if (data.match) {
          resDiv.className = 'result-card result-success';
          document.getElementById('resultTitle').innerText = `✅ Rostro Autorizado: ${data.name}`;
          document.getElementById('resultDetails').innerHTML = `
            Similitud biométrica: <strong>${(data.confidence * 100).toFixed(1)}%</strong><br>
            Acción en Home Assistant: <strong>Abrir puerta (open_door)</strong><br>
            Tiempo inferencia CPU: ${data.execution_time_ms} ms
          `;
        } else if (data.detected) {
          resDiv.className = 'result-card result-fail';
          document.getElementById('resultTitle').innerText = `❌ No Autorizado`;
          document.getElementById('resultDetails').innerHTML = `
            Mejor candidato: ${data.candidate_name || 'Desconocido'} (${(data.confidence * 100).toFixed(1)}%)<br>
            Umbral requerido: ${(data.threshold_used * 100).toFixed(1)}%<br>
            Tiempo inferencia CPU: ${data.execution_time_ms} ms
          `;
        } else {
          resDiv.className = 'result-card result-fail';
          document.getElementById('resultTitle').innerText = `⚠️ No se detectó rostro`;
          document.getElementById('resultDetails').innerText = `Verifica que la cara esté bien iluminada y frente a la cámara.`;
        }
      } catch (err) {
        resDiv.className = 'result-card result-fail';
        document.getElementById('resultTitle').innerText = 'Error al analizar';
        document.getElementById('resultDetails').innerText = err.toString();
      }
    }

    // -------------------------------------------------------------------------
    // 3-Day History & Self-Training UI
    // -------------------------------------------------------------------------
    async function loadHistoryCount() {
      try {
        const res = await fetch('/api/v1/history');
        if (res.ok) {
          const list = await res.json();
          document.getElementById('historyCountBadge').innerText = list.length;
        }
      } catch (err) {}
    }

    async function loadHistory() {
      const container = document.getElementById('historyContainer');
      container.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">Cargando capturas...</p>';
      
      try {
        // Ensure people list is fresh for options dropdown
        await loadPeople();
        const res = await fetch('/api/v1/history');
        if (res.status === 401) { showLoginView(); return; }
        const list = await res.json();
        document.getElementById('historyCountBadge').innerText = list.length;

        if (!list || list.length === 0) {
          container.innerHTML = `
            <div style="text-align:center; padding:3rem 1rem; color:var(--text-muted);">
              <span style="font-size:2.5rem; display:block; margin-bottom:0.5rem;">📷</span>
              No hay capturas registradas en los últimos 3 días.<br>
              Aparecerán automáticamente aquí cada vez que Fermax o Home Assistant envíe una imagen al microservicio.
            </div>
          `;
          return;
        }

        const optionsHtml = registeredPeopleList.map(p => `<option value="${p.name}">${p.name}</option>`).join('');

        container.innerHTML = `
          <div class="history-grid">
            ${list.map(c => `
              <div class="history-card" id="card_${c.id}">
                <div class="history-img-wrap">
                  <img class="history-img" src="/api/v1/history/image/${c.filename}" alt="Captura" loading="lazy">
                  <div class="history-badge ${c.match ? 'badge-match' : 'badge-no-match'}">
                    ${c.match ? '✅ ' + c.matched_name : '❌ ' + (c.candidate_name || 'Desconocido')}
                  </div>
                </div>
                <div class="history-body">
                  <div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <span style="font-size:0.75rem; color:var(--text-muted);">🕒 ${c.created_at}</span>
                      <span style="font-size:0.75rem; color:var(--primary); font-weight:bold;">${(c.confidence * 100).toFixed(0)}% sim</span>
                    </div>
                  </div>
                  
                  <div style="margin-top:0.75rem; border-top:1px solid var(--border); padding-top:0.6rem;">
                    ${c.assigned_to ? `
                      <div style="font-size:0.8rem; color:var(--accent); font-weight:bold; margin-bottom:0.4rem;">
                        ✓ Autoentrenado en: ${c.assigned_to}
                      </div>
                      <button class="danger" style="width:100%; padding:0.3rem;" onclick="deleteHistoryCapture('${c.id}')">🗑️ Borrar del Historial</button>
                    ` : `
                      <label style="font-size:0.75rem; margin-bottom:0.2rem;">¿A quién pertenece?</label>
                      <select id="sel_${c.id}" style="margin-bottom:0.4rem; padding:0.4rem 0.6rem; font-size:0.8rem;">
                        <option value="">-- Seleccionar Persona --</option>
                        ${optionsHtml}
                      </select>
                      <div style="display:flex; gap:0.35rem;">
                        <button style="flex:1; padding:0.4rem; font-size:0.78rem;" onclick="assignCapture('${c.id}')">
                          ➕ Autoentrenar
                        </button>
                        <button class="danger" style="padding:0.4rem 0.6rem;" title="Eliminar captura" onclick="deleteHistoryCapture('${c.id}')">
                          🗑️
                        </button>
                      </div>
                    `}
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        `;
      } catch (err) {
        container.innerHTML = `<p style="color:var(--danger); font-size:0.85rem;">Error cargando historial: ${err}</p>`;
      }
    }

    async function assignCapture(captureId) {
      const select = document.getElementById(`sel_${captureId}`);
      let personName = select ? select.value : '';

      if (!personName) {
        personName = prompt('Introduce el nombre de la persona a la que asociar esta foto:');
        if (!personName || !personName.trim()) return;
        personName = personName.trim();
      }

      try {
        const res = await fetch('/api/v1/history/assign', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ capture_id: captureId, person_name: personName })
        });
        const data = await res.json();
        if (res.ok) {
          alert(`¡Autoentrenamiento completado! ${data.message}`);
          loadHistory();
          loadPeople();
        } else {
          alert('Error: ' + (data.detail || 'No se pudo asignar la foto'));
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    async function deleteHistoryCapture(captureId) {
      if (!confirm('¿Eliminar esta foto del histórico?')) return;
      try {
        const res = await fetch(`/api/v1/history/${encodeURIComponent(captureId)}`, { method: 'DELETE' });
        if (res.ok) {
          loadHistory();
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    async function clearAllHistory() {
      if (!confirm('¿Estás seguro de que deseas eliminar todas las capturas del historial de 3 días?')) return;
      try {
        const res = await fetch('/api/v1/history', { method: 'DELETE' });
        if (res.ok) {
          loadHistory();
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    // -------------------------------------------------------------------------
    // API Tokens Management
    // -------------------------------------------------------------------------
    async function loadTokens() {
      try {
        const res = await fetch('/api/v1/tokens');
        if (res.status === 401) return;
        const tokens = await res.json();
        const tbody = document.getElementById('tokensTableBody');
        
        if (!tokens || tokens.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:1rem;">No hay tokens activos. Genera uno para Home Assistant con el botón superior.</td></tr>';
          return;
        }

        tbody.innerHTML = tokens.map(t => `
          <tr>
            <td><strong>${t.name}</strong></td>
            <td><span class="token-code">${t.prefix}</span></td>
            <td>${t.created_at || '-'}</td>
            <td>${t.last_used ? `<span style="color:var(--accent);">✓ ${t.last_used}</span>` : '<span style="color:var(--text-muted);">Nunca</span>'}</td>
            <td style="text-align:right;">
              <button class="danger" title="Revocar token" onclick="deleteToken('${t.id}', '${t.name}')">🗑️ Revocar</button>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.error('Error loading tokens:', err);
      }
    }

    async function promptCreateToken() {
      const name = prompt('Introduce un nombre para identificar este Token (ej: "Home Assistant Fermax"):');
      if (!name || !name.trim()) return;

      try {
        const res = await fetch('/api/v1/tokens', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim() })
        });
        const data = await res.json();
        if (res.ok) {
          lastGeneratedSecret = data.token;
          document.getElementById('rawTokenDisplay').innerText = data.token;
          document.getElementById('tokenModal').style.display = 'flex';
          loadTokens();
        } else {
          alert('Error: ' + (data.detail || 'No se pudo generar el token'));
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    function copyGeneratedToken() {
      if (!lastGeneratedSecret) return;
      navigator.clipboard.writeText(lastGeneratedSecret).then(() => {
        alert('¡Token copiado al portapapeles!');
      });
    }

    function closeTokenModal() {
      document.getElementById('tokenModal').style.display = 'none';
      lastGeneratedSecret = '';
      document.getElementById('rawTokenDisplay').innerText = '';
    }

    async function deleteToken(id, name) {
      if (!confirm(`¿Estás seguro de que deseas revocar y eliminar el token "${name}"? Los dispositivos que lo utilicen perderán el acceso.`)) return;
      await fetch(`/api/v1/tokens/${encodeURIComponent(id)}`, { method: 'DELETE' });
      loadTokens();
    }

    // Init
    setupDragAndDrop('dropZoneNew', 'newPhotos', updateNewPhotoCount);
    setupDragAndDrop('dropZoneTest', 'testPhoto', runTest);
    checkAuth();
  </script>
</body>
</html>"""
