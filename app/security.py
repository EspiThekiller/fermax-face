"""Security and Authentication module for Fermax Face Recognition Service.

Provides:
- First-boot random 10-character admin password generation with one-time logging.
- Mandatory replacement / deletion of the initial 'admin' user on first login.
- Named API Tokens with SHA-256 storage (shown only once upon creation, revocable).
- Session management and brute-force protection.
- Rejection of tokens injected via environment variables.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import string
import time
import uuid
from typing import Any

_LOGGER = logging.getLogger("fermax_security")


class SecurityManager:
    """Manages persistent credentials, API tokens, and web sessions."""

    def __init__(self, data_dir: str | None = None) -> None:
        if not data_dir:
            data_dir = os.getenv("DATA_DIR", "/app/data")
        self.data_dir = data_dir
        self.db_path = os.path.join(self.data_dir, "security.json")

        self.data: dict[str, Any] = {
            "version": 1,
            "first_boot_completed": False,
            "users": {},
            "tokens": [],
            "sessions": {},
        }
        self.failed_login_attempts: dict[str, list[float]] = {}
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Create storage directory and load or initialize security DB."""
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
                _LOGGER.info(
                    "Security database loaded from %s (%d user(s), %d active token(s)).",
                    self.db_path,
                    len(self.data.get("users", {})),
                    len(self.data.get("tokens", [])),
                )
            except Exception as exc:
                _LOGGER.error("Failed to parse %s: %s. Re-initializing...", self.db_path, exc)
                self._initialize_first_boot()
        else:
            self._initialize_first_boot()

    def _initialize_first_boot(self) -> None:
        """Generate random 10-character admin password on first boot and print to log once."""
        # Generate 10 random alphanumeric characters
        alphabet = string.ascii_letters + string.digits
        initial_password = "".join(secrets.choice(alphabet) for _ in range(10))

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(initial_password, salt)

        self.data["first_boot_completed"] = True
        self.data["users"] = {
            "admin": {
                "salt": salt,
                "hash": pw_hash,
                "must_change": True,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        self.data["tokens"] = []
        self.data["sessions"] = {}
        self._save()

        # Print conspicuous one-time security banner to logs
        border = "=" * 76
        _LOGGER.warning("\n%s", border)
        _LOGGER.warning("🔐 [SEGURIDAD] FERMAX FACE SERVICE - PRIMER INICIO DETECTADO")
        _LOGGER.warning("-" * 76)
        _LOGGER.warning("Se ha generado la cuenta inicial temporal de administrador:")
        _LOGGER.warning("  USUARIO:    admin")
        _LOGGER.warning("  CONTRASEÑA: %s", initial_password)
        _LOGGER.warning("")
        _LOGGER.warning("AVISO IMPORTANTE:")
        _LOGGER.warning("1. Esta contraseña solo se mostrará en este registro del primer arranque.")
        _LOGGER.warning("2. En los siguientes inicios no volverá a aparecer en ningún log.")
        _LOGGER.warning("3. Al iniciar sesión por primera vez en el panel web, el sistema te")
        _LOGGER.warning("   obligará a ELIMINAR el usuario 'admin' y configurar tus nuevas")
        _LOGGER.warning("   credenciales de administrador definitivas.")
        _LOGGER.warning("%s\n", border)

    def _save(self) -> None:
        """Atomic save of security database to disk."""
        tmp_path = f"{self.db_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_path, self.db_path)
        except Exception as exc:
            _LOGGER.error("Error saving security data to %s: %s", self.db_path, exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """PBKDF2-HMAC-SHA256 password hash (100,000 iterations)."""
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
        ).hex()

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is currently rate limited due to repeated failures."""
        now = time.time()
        attempts = self.failed_login_attempts.get(client_ip, [])
        # Keep attempts from last 5 minutes
        recent = [t for t in attempts if now - t < 300]
        self.failed_login_attempts[client_ip] = recent
        if len(recent) >= 5:
            # Cooldown of 60s from the latest attempt
            if now - recent[-1] < 60:
                return True
        return False

    def register_failed_attempt(self, client_ip: str) -> None:
        """Record a failed login attempt for rate limiting."""
        now = time.time()
        if client_ip not in self.failed_login_attempts:
            self.failed_login_attempts[client_ip] = []
        self.failed_login_attempts[client_ip].append(now)

    def reset_failed_attempts(self, client_ip: str) -> None:
        """Reset failed attempts upon successful authentication."""
        self.failed_login_attempts.pop(client_ip, None)

    def authenticate_user(
        self, username: str, password: str, client_ip: str = "unknown"
    ) -> tuple[bool, str | None, bool]:
        """Authenticate user credentials.
        
        Returns:
            (success: bool, username: str | None, must_change: bool)
        """
        if self.is_rate_limited(client_ip):
            _LOGGER.warning("Rate limit exceeded for IP: %s", client_ip)
            return False, None, False

        users = self.data.get("users", {})
        user_data = users.get(username)
        if not user_data:
            self.register_failed_attempt(client_ip)
            return False, None, False

        salt = user_data.get("salt", "")
        expected_hash = user_data.get("hash", "")
        computed_hash = self._hash_password(password, salt)

        if not hmac.compare_digest(computed_hash, expected_hash):
            self.register_failed_attempt(client_ip)
            return False, None, False

        self.reset_failed_attempts(client_ip)
        must_change = bool(user_data.get("must_change", False))
        return True, username, must_change

    def setup_new_admin(self, new_username: str, new_password: str) -> bool:
        """Mandatory initial transition: eliminate temporary 'admin' user and configure new permanent admin."""
        new_username = new_username.strip()
        new_password = new_password.strip()

        if len(new_username) < 3 or len(new_password) < 6:
            raise ValueError("El usuario debe tener al menos 3 caracteres y la contraseña al menos 6.")

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(new_password, salt)

        # Eliminate temporary 'admin' user completely
        users = self.data.get("users", {})
        users.pop("admin", None)

        users[new_username] = {
            "salt": salt,
            "hash": pw_hash,
            "must_change": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.data["users"] = users
        # Invalidate old sessions
        self.data["sessions"] = {}
        self._save()

        _LOGGER.info(
            "Usuario inicial temporal 'admin' eliminado permanentemente. Nuevo administrador '%s' configurado.",
            new_username,
        )
        return True

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change password for an existing user."""
        success, _, _ = self.authenticate_user(username, old_password)
        if not success:
            return False

        if len(new_password.strip()) < 6:
            raise ValueError("La nueva contraseña debe tener al menos 6 caracteres.")

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(new_password.strip(), salt)

        self.data["users"][username]["salt"] = salt
        self.data["users"][username]["hash"] = pw_hash
        self.data["users"][username]["must_change"] = False
        self._save()
        return True

    def create_session(self, username: str) -> str:
        """Create a new web session for username valid for 7 days."""
        session_id = secrets.token_hex(24)
        expires_at = time.time() + (7 * 24 * 3600)  # 7 days
        must_change = bool(self.data.get("users", {}).get(username, {}).get("must_change", False))

        if "sessions" not in self.data:
            self.data["sessions"] = {}

        self.data["sessions"][session_id] = {
            "username": username,
            "expires_at": expires_at,
            "must_change": must_change,
        }
        self._save()
        return session_id

    def verify_session(self, session_id: str | None) -> tuple[bool, str | None, bool]:
        """Validate an active web session.
        
        Returns:
            (is_valid: bool, username: str | None, must_change: bool)
        """
        if not session_id:
            return False, None, False

        sessions = self.data.get("sessions", {})
        info = sessions.get(session_id)
        if not info:
            return False, None, False

        if time.time() > info.get("expires_at", 0):
            # Session expired
            sessions.pop(session_id, None)
            self._save()
            return False, None, False

        username = info.get("username")
        user_info = self.data.get("users", {}).get(username, {})
        must_change = bool(user_info.get("must_change", False))

        return True, username, must_change

    def delete_session(self, session_id: str | None) -> None:
        """Log out / invalidate session."""
        if not session_id:
            return
        if "sessions" in self.data and session_id in self.data["sessions"]:
            self.data["sessions"].pop(session_id, None)
            self._save()

    def create_api_token(self, name: str) -> tuple[dict[str, Any], str]:
        """Generate a new named API token.
        
        The plaintext token is returned ONCE to display to the user.
        Only the SHA-256 hash is saved in storage.
        """
        token_name = name.strip() or "API Token"
        raw_secret = f"fft_{secrets.token_urlsafe(32)}"
        token_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
        token_id = str(uuid.uuid4())
        prefix = f"{raw_secret[:9]}...{raw_secret[-4:]}"

        token_record = {
            "id": token_id,
            "name": token_name,
            "prefix": prefix,
            "hash": token_hash,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_used": None,
        }

        if "tokens" not in self.data:
            self.data["tokens"] = []

        self.data["tokens"].append(token_record)
        self._save()

        # Public metadata (does not include hash)
        metadata = {
            "id": token_id,
            "name": token_name,
            "prefix": prefix,
            "created_at": token_record["created_at"],
            "last_used": None,
        }
        return metadata, raw_secret

    def list_api_tokens(self) -> list[dict[str, Any]]:
        """List all active API tokens with metadata. Plaintext tokens cannot be retrieved."""
        tokens = self.data.get("tokens", [])
        return [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "prefix": t.get("prefix"),
                "created_at": t.get("created_at"),
                "last_used": t.get("last_used"),
            }
            for t in tokens
        ]

    def delete_api_token(self, token_id: str) -> bool:
        """Revoke and delete an API token by ID."""
        tokens = self.data.get("tokens", [])
        initial_len = len(tokens)
        self.data["tokens"] = [t for t in tokens if t.get("id") != token_id]
        if len(self.data["tokens"]) != initial_len:
            self._save()
            return True
        return False

    def verify_api_token(self, raw_token: str | None) -> bool:
        """Verify an incoming API token against stored SHA-256 hashes."""
        if not raw_token:
            return False

        token_hash = hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()
        tokens = self.data.get("tokens", [])
        matched = False

        for t in tokens:
            if hmac.compare_digest(token_hash, t.get("hash", "")):
                matched = True
                t["last_used"] = time.strftime("%Y-%m-%d %H:%M:%S")
                break

        if matched:
            self._save()
            return True

        return False
