"""Capture history and self-training manager for Fermax Face Recognition."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import FaceRecognitionEngine

_LOGGER = logging.getLogger("fermax_history")

# Default retention: 3 days in seconds (3 * 24 * 3600 = 259200)
RETENTION_SECONDS = 3 * 24 * 3600


class HistoryManager:
    """Manages 3-day history of verified captures and self-training assignment."""

    def __init__(self, data_dir: str = "/app/data", retention_seconds: int = RETENTION_SECONDS) -> None:
        self.data_dir = data_dir
        self.history_dir = os.path.join(data_dir, "history")
        self.metadata_file = os.path.join(self.history_dir, "history.json")
        self.retention_seconds = retention_seconds
        self._entries: list[dict[str, Any]] = []

        os.makedirs(self.history_dir, exist_ok=True)
        self._load()
        self.cleanup_expired()

    def _load(self) -> None:
        """Load history metadata from disk."""
        if not os.path.isfile(self.metadata_file):
            self._entries = []
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._entries = data
                else:
                    self._entries = []
        except Exception as exc:
            _LOGGER.warning("Could not read history file %s: %s", self.metadata_file, exc)
            self._entries = []

    def _save(self) -> None:
        """Atomically persist history metadata to disk."""
        tmp_file = f"{self.metadata_file}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self.metadata_file)
        except Exception as exc:
            _LOGGER.error("Failed to save history metadata: %s", exc)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def cleanup_expired(self) -> int:
        """Remove captures older than 3 days from disk and database."""
        now = time.time()
        cutoff = now - self.retention_seconds
        retained: list[dict[str, Any]] = []
        removed_count = 0

        for entry in self._entries:
            ts = entry.get("timestamp_epoch", 0.0)
            filename = entry.get("filename")
            if ts < cutoff:
                # Expired (> 3 days old)
                if filename:
                    img_path = os.path.join(self.history_dir, filename)
                    if os.path.isfile(img_path):
                        try:
                            os.remove(img_path)
                        except OSError as err:
                            _LOGGER.warning("Could not delete expired capture image %s: %s", img_path, err)
                removed_count += 1
            else:
                retained.append(entry)

        if removed_count > 0:
            self._entries = retained
            self._save()
            _LOGGER.info("History cleanup: removed %d captures older than 3 days.", removed_count)

        return removed_count

    def record_capture(self, image_bytes: bytes, recognition_result: dict[str, Any]) -> dict[str, Any] | None:
        """Save a capture verified/processed by the API and register in the 3-day history."""
        # Run periodic cleanup
        self.cleanup_expired()

        cap_id = f"cap_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        filename = f"{cap_id}.jpg"
        filepath = os.path.join(self.history_dir, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
        except Exception as exc:
            _LOGGER.error("Failed to save capture image %s: %s", filepath, exc)
            return None

        now_dt = datetime.now()
        entry: dict[str, Any] = {
            "id": cap_id,
            "filename": filename,
            "created_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_epoch": now_dt.timestamp(),
            "detected": bool(recognition_result.get("detected", False)),
            "match": bool(recognition_result.get("match", False)),
            "matched_name": str(recognition_result.get("name", "Desconocido")),
            "candidate_name": str(recognition_result.get("candidate_name", recognition_result.get("name", "Desconocido"))),
            "confidence": float(recognition_result.get("confidence", 0.0)),
            "det_score": float(recognition_result.get("det_score", 0.0)),
            "execution_time_ms": float(recognition_result.get("execution_time_ms", 0.0)),
            "assigned_to": None,
        }

        self._entries.insert(0, entry)
        self._save()
        _LOGGER.info(
            "Capture recorded in history: %s (name=%s, conf=%.2f)",
            cap_id,
            entry["candidate_name"],
            entry["confidence"],
        )
        return entry

    def list_captures(self) -> list[dict[str, Any]]:
        """Return all valid captures within the 3-day retention period, newest first."""
        self.cleanup_expired()
        return self._entries

    def get_image_path(self, filename: str) -> str | None:
        """Get absolute path to an image if it exists within history."""
        # Sanitize filename
        safe_name = os.path.basename(filename)
        path = os.path.join(self.history_dir, safe_name)
        if os.path.isfile(path):
            return path
        return None

    def assign_to_person(
        self,
        capture_id: str,
        person_name: str,
        engine: FaceRecognitionEngine,
    ) -> dict[str, Any]:
        """Assign a captured photo to a person to retrain and improve accuracy."""
        clean_name = person_name.strip().replace(" ", "_")
        if not clean_name:
            return {"status": "error", "message": "Nombre de persona no válido"}

        entry = next((e for e in self._entries if e.get("id") == capture_id), None)
        if not entry:
            return {"status": "error", "message": "Captura no encontrada en el historial"}

        filename = entry.get("filename")
        if not filename:
            return {"status": "error", "message": "Archivo de captura ausente"}

        img_path = os.path.join(self.history_dir, filename)
        if not os.path.isfile(img_path):
            return {"status": "error", "message": "El archivo de imagen físico no existe"}

        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
        except Exception as exc:
            return {"status": "error", "message": f"Error leyendo imagen: {exc}"}

        # Register photo in person's dataset and recalculate embeddings
        dest_filename = f"trained_{capture_id}.jpg"
        reg_result = engine.register_face(name=clean_name, image_bytes=img_bytes, filename=dest_filename)

        if reg_result.get("status") == "error":
            return reg_result

        # Mark capture as assigned
        entry["assigned_to"] = clean_name
        self._save()

        _LOGGER.info(
            "Self-training: Capture %s assigned to '%s' (total photos now: %d)",
            capture_id,
            clean_name,
            reg_result.get("photos_count", 0),
        )

        return {
            "status": "ok",
            "message": f"Foto incorporada con éxito al perfil biométrico de '{clean_name}'.",
            "capture_id": capture_id,
            "person": clean_name,
            "photos_count": reg_result.get("photos_count", 0),
        }

    def delete_capture(self, capture_id: str) -> bool:
        """Remove a capture from history and delete its file."""
        entry = next((e for e in self._entries if e.get("id") == capture_id), None)
        if not entry:
            return False

        filename = entry.get("filename")
        if filename:
            img_path = os.path.join(self.history_dir, filename)
            if os.path.isfile(img_path):
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        self._entries = [e for e in self._entries if e.get("id") != capture_id]
        self._save()
        return True

    def clear_all(self) -> int:
        """Clear all captures from history and remove image files."""
        count = len(self._entries)
        for entry in self._entries:
            fn = entry.get("filename")
            if fn:
                p = os.path.join(self.history_dir, fn)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        self._entries = []
        self._save()
        return count
