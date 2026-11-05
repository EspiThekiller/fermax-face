"""Face recognition engine powered by InsightFace and ONNX Runtime (CPU)."""

from __future__ import annotations

import io
import logging
import os
import shutil
import time
import warnings
from typing import Any

# Suppress internal library deprecation warnings from skimage / insightface
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from PIL import Image

_LOGGER = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """High-accuracy, lightweight face recognition engine optimized for CPU."""

    def __init__(
        self,
        known_faces_dir: str = "/app/known_faces",
        model_name: str = "buffalo_s",
        det_size: int = 320,
        similarity_threshold: float = 0.65,
    ) -> None:
        self.known_faces_dir = known_faces_dir
        self.model_name = model_name
        self.det_size = (det_size, det_size)
        self.similarity_threshold = similarity_threshold

        self.app: Any = None
        self.known_db: dict[str, list[np.ndarray]] = {}
        os.makedirs(self.known_faces_dir, exist_ok=True)

    def initialize(self) -> None:
        """Initialize the FaceAnalysis ONNX models."""
        from insightface.app import FaceAnalysis

        _LOGGER.info(
            "Initializing InsightFace '%s' engine with CPUExecutionProvider...",
            self.model_name,
        )
        self.app = FaceAnalysis(
            name=self.model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)
        _LOGGER.info("InsightFace models loaded successfully.")
        self.load_known_faces()

    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        """Decode image bytes into BGR numpy array."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

        # Fallback using PIL
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            rgb_arr = np.array(pil_img)
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            _LOGGER.error("Failed to decode image: %s", exc)
            return None

    def load_known_faces(self) -> None:
        """Scan known_faces_dir and compute normalized embeddings for each person."""
        self.known_db = {}
        if not os.path.isdir(self.known_faces_dir):
            return

        total_photos = 0
        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            person_embeddings: list[np.ndarray] = []
            for file_name in os.listdir(person_dir):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    file_path = os.path.join(person_dir, file_name)
                    try:
                        with open(file_path, "rb") as f:
                            img_bytes = f.read()
                        img = self._decode_image(img_bytes)
                        if img is None:
                            continue
                        faces = self.app.get(img)
                        if faces:
                            # Use highest confidence face
                            best_face = max(faces, key=lambda f: f.det_score)
                            emb = best_face.normed_embedding
                            person_embeddings.append(emb)
                            total_photos += 1
                    except Exception as exc:
                        _LOGGER.warning("Could not process %s: %s", file_path, exc)

            if person_embeddings:
                self.known_db[person_name] = person_embeddings
                _LOGGER.info(
                    "Person loaded: %s (%d photos registered)",
                    person_name,
                    len(person_embeddings),
                )

        _LOGGER.info(
            "Known faces database ready: %d people, %d embeddings.",
            len(self.known_db),
            total_photos,
        )

    def recognize(
        self,
        image_bytes: bytes,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detect face in query image and match against known faces."""
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self.similarity_threshold

        img = self._decode_image(image_bytes)
        if img is None:
            return {
                "detected": False,
                "match": False,
                "name": "Error de imagen",
                "confidence": 0.0,
                "box": None,
                "faces_count": 0,
                "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        faces = self.app.get(img)
        t_detect = time.perf_counter()

        if not faces:
            return {
                "detected": False,
                "match": False,
                "name": "Nadie detectado",
                "confidence": 0.0,
                "box": None,
                "faces_count": 0,
                "execution_time_ms": round((t_detect - t0) * 1000, 2),
            }

        # Select the most prominent face (largest bounding box area)
        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        query_emb = best_face.normed_embedding

        box = [int(v) for v in best_face.bbox]  # [x1, y1, x2, y2]
        det_score = float(best_face.det_score)

        best_name = "Desconocido"
        best_sim = 0.0

        for person_name, embeddings in self.known_db.items():
            for emb in embeddings:
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = person_name

        matched = best_sim >= thresh
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "detected": True,
            "match": matched,
            "name": best_name if matched else "Desconocido",
            "candidate_name": best_name,
            "confidence": round(best_sim, 4),
            "det_score": round(det_score, 4),
            "threshold_used": thresh,
            "box": box,
            "faces_count": len(faces),
            "execution_time_ms": total_time_ms,
        }

    def register_face(
        self,
        name: str,
        image_bytes: bytes,
        filename: str = "photo.jpg",
    ) -> dict[str, Any]:
        """Save a new photo for a person and update in-memory embeddings."""
        clean_name = name.strip().replace(" ", "_")
        person_dir = os.path.join(self.known_faces_dir, clean_name)
        os.makedirs(person_dir, exist_ok=True)

        img = self._decode_image(image_bytes)
        if img is None:
            return {"status": "error", "message": "Formato de imagen inválido"}

        faces = self.app.get(img)
        if not faces:
            return {"status": "error", "message": "No se detectó ningún rostro en la foto"}

        best_face = max(faces, key=lambda f: f.det_score)
        emb = best_face.normed_embedding

        if clean_name not in self.known_db:
            self.known_db[clean_name] = []
        self.known_db[clean_name].append(emb)

        # Save photo file
        save_path = os.path.join(person_dir, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        return {
            "status": "ok",
            "name": clean_name,
            "photos_count": len(self.known_db[clean_name]),
            "det_score": float(best_face.det_score),
        }

    def register_faces_bulk(
        self,
        name: str,
        photos: list[tuple[bytes, str]],
    ) -> dict[str, Any]:
        """Save multiple photos for a person and update in-memory embeddings."""
        clean_name = name.strip().replace(" ", "_")
        person_dir = os.path.join(self.known_faces_dir, clean_name)
        os.makedirs(person_dir, exist_ok=True)

        if clean_name not in self.known_db:
            self.known_db[clean_name] = []

        added = 0
        failed = 0

        for img_bytes, filename in photos:
            img = self._decode_image(img_bytes)
            if img is None:
                failed += 1
                continue
            faces = self.app.get(img)
            if not faces:
                failed += 1
                continue

            best_face = max(faces, key=lambda f: f.det_score)
            emb = best_face.normed_embedding
            self.known_db[clean_name].append(emb)

            save_path = os.path.join(person_dir, filename)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            added += 1

        return {
            "status": "ok",
            "name": clean_name,
            "added_photos": added,
            "failed_photos": failed,
            "total_photos": len(self.known_db[clean_name]),
        }

    def delete_person(self, name: str) -> bool:
        """Remove a person and their photos."""
        clean_name = name.strip().replace(" ", "_")
        person_dir = os.path.join(self.known_faces_dir, clean_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir, ignore_errors=True)
        if clean_name in self.known_db:
            del self.known_db[clean_name]
            return True
        return False

    def list_people(self) -> list[dict[str, Any]]:
        """List all registered people and photo counts."""
        return [
            {"name": name, "photos_count": len(embs)}
            for name, embs in self.known_db.items()
        ]
