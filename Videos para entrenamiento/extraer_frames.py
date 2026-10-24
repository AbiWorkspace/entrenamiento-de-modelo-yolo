from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "Falta la dependencia 'opencv-python-headless'. Instálala con pip.\n"
    )
    raise SystemExit(1) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("Falta la dependencia 'numpy'. Instálala con pip.\n")
    raise SystemExit(1) from exc

LOGGER = logging.getLogger("extraer_frames")


@dataclass(frozen=True)
class ExtractConfig:
    """Hiperparámetros del extractor de keyframes (Fase 1, Plan.md)."""

    max_frames_por_video: int = 45
    workers: int = 8
    metric_size: tuple[int, int] = (320, 320)
    jpeg_quality: int = 95
    kinematic_fraction: float = 0.70
    blur_fraction: float = 0.30
    diff_threshold_kine: float = 2.0
    min_separation_kine: int = 5
    diff_threshold_blur: float = 1.5
    blur_pct_low: float = 0.15
    blur_pct_high: float = 0.40

    def buckets(self) -> tuple[int, int]:
        total = self.max_frames_por_video
        kine = max(1, round(total * self.kinematic_fraction))
        blur = max(1, total - kine)
        return kine, blur


@dataclass(frozen=True)
class VideoResult:
    """Resultado de la extracción de un único video."""

    video_id: str
    nombre_video: str
    total_frames: int
    extraidos_kine: int
    extraidos_blur: int
    ruta_salida: Path


@dataclass(frozen=True)
class FrameMetric:
    """Métricas cinemáticas y de nitidez de un frame (resolución reducida)."""

    idx: int
    diff_score: float
    blur_score: float


def _config_log() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _semillas() -> None:
    random.seed(42)
    np.random.seed(42)


def _video_id(ruta_video: Path) -> str:
    """Prefijo único e inmutable por video: vid_{hash_corto}_{nombre_limpio}."""
    digest = hashlib.sha1(str(ruta_video).encode("utf-8")).hexdigest()[:8]
    base = ruta_video.stem
    base = re.sub(r"\s*\(.*?\)\s*$", "", base)
    base = re.sub(r"~\d+\s*$", "", base)
    limpio = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_").lower()
    if not limpio:
        limpio = "video"
    return f"vid_{digest}_{limpio}"


def _leer_metricas(
    ruta_video: Path,
    cfg: ExtractConfig,
) -> list[FrameMetric]:
    """Pase 1 (streaming): calcula diff vs frame consecutivo + nitidez por frame.

    El diff se mide contra el frame ANTERIOR inmediato (no el último elegido),
    así la métrica refleja movimiento instantáneo y es determinista/paralelizable.
    La selección de keyframes se hace después en _seleccionar_kine/_seleccionar_blur.
    """
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        cap.release()
        return []

    w, h = cfg.metric_size
    metricas: list[FrameMetric] = []
    gray_prev: Optional[np.ndarray] = None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        diff = (
            float(np.mean(cv2.absdiff(gray, gray_prev)))
            if gray_prev is not None
            else 0.0
        )
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metricas.append(FrameMetric(idx=idx, diff_score=diff, blur_score=blur))
        gray_prev = gray
        idx += 1

    cap.release()
    return metricas


def _seleccionar_kine(
    metricas: list[FrameMetric],
    cfg: ExtractConfig,
) -> list[int]:
    """Cubeta cinemática normal (70% del cupo): diff >= umbral y separación mínima."""
    n_kine, _ = cfg.buckets()
    kine_indices: list[int] = []
    ultimo_elegido = -(10**9)
    for m in metricas:
        if len(kine_indices) >= n_kine:
            break
        if (
            m.diff_score >= cfg.diff_threshold_kine
            and (m.idx - ultimo_elegido) >= cfg.min_separation_kine
        ):
            kine_indices.append(m.idx)
            ultimo_elegido = m.idx

    return kine_indices


def _seleccionar_blur(
    metricas: list[FrameMetric],
    kine_indices: list[int],
    cfg: ExtractConfig,
) -> list[int]:
    """Cubeta de alto movimiento / motion blur (30% del cupo).

    Frames con diff_score >= umbral y nitidez dentro del percentil [p15, p40] del
    propio video (se descarta el 15% inferior por pérdida total de imagen).
    """
    _, n_blur = cfg.buckets()
    blur_indices: list[int] = []
    if n_blur > 0 and metricas:
        valores_blur = np.array([m.blur_score for m in metricas], dtype=np.float64)
        p_low = float(np.percentile(valores_blur, cfg.blur_pct_low * 100.0))
        p_high = float(np.percentile(valores_blur, cfg.blur_pct_high * 100.0))

        kine_set = set(kine_indices)
        ultimo_elegido = -(10**9)
        for m in sorted(metricas, key=lambda x: -x.diff_score):
            if len(blur_indices) >= n_blur:
                break
            if m.idx in kine_set:
                continue
            if not (p_low <= m.blur_score <= p_high):
                continue
            if m.diff_score < cfg.diff_threshold_blur:
                continue
            if (m.idx - ultimo_elegido) < 3:
                continue
            blur_indices.append(m.idx)
            ultimo_elegido = m.idx

    return blur_indices


def _guardar_frames(
    ruta_video: Path,
    indices: list[int],
    video_id: str,
    carpeta_salida: Path,
    cfg: ExtractConfig,
) -> int:
    """Pase 2: reubica a cada índice seleccionado y guarda el frame en resolución original."""
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        cap.release()
        return 0

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
    guardados = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            LOGGER.warning(
                "Frame %d no reubicable en %s (se omite).", idx, ruta_video.name
            )
            continue
        nombre = f"{video_id}_f{idx:05d}.jpg"
        ruta_salida = carpeta_salida / nombre
        if cv2.imwrite(str(ruta_salida), frame, params):
            guardados += 1
    cap.release()
    return guardados


def _procesar_video(
    ruta_video: Path,
    carpeta_salida: Path,
    cfg: ExtractConfig,
) -> VideoResult:
    """Tarea paralela: extracción completa de un video (pase 1 + selección + pase 2)."""
    vid = _video_id(ruta_video)
    metricas = _leer_metricas(ruta_video, cfg)
    if not metricas:
        return VideoResult(
            video_id=vid,
            nombre_video=ruta_video.name,
            total_frames=0,
            extraidos_kine=0,
            extraidos_blur=0,
            ruta_salida=carpeta_salida,
        )

    kine_indices = _seleccionar_kine(metricas, cfg)
    blur_indices = _seleccionar_blur(metricas, kine_indices, cfg)

    guardados = _guardar_frames(
        ruta_video, kine_indices + blur_indices, vid, carpeta_salida, cfg
    )
    n_kine = sum(
        1 for i in kine_indices if (carpeta_salida / f"{vid}_f{i:05d}.jpg").exists()
    )
    n_blur = guardados - n_kine

    return VideoResult(
        video_id=vid,
        nombre_video=ruta_video.name,
        total_frames=len(metricas),
        extraidos_kine=n_kine,
        extraidos_blur=n_blur,
        ruta_salida=carpeta_salida,
    )


def _archivar_legado(carpeta_salida: Path) -> Optional[Path]:
    """Mueve los frames antiguos (frame_NNNNNN.jpg) a un subdirectorio backup (no destructivo)."""
    antiguos = sorted(
        p
        for p in carpeta_salida.iterdir()
        if p.name.startswith("frame_") and p.suffix.lower() == ".jpg"
    )
    if not antiguos:
        return None
    backup = carpeta_salida / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=False)
    for p in antiguos:
        destino = backup / p.name
        if not destino.exists() and p.stat().st_size > 0:
            p.rename(destino)
    LOGGER.info("Archivadas %d imágenes antiguas en %s", len(antiguos), backup)
    return backup


def _descubrir_videos(carpeta_videos: Path) -> list[Path]:
    formatos = {".mp4", ".mov", ".avi", ".mkv"}
    videos = [
        p
        for p in carpeta_videos.iterdir()
        if p.is_file() and p.suffix.lower() in formatos
    ]
    return sorted(videos, key=lambda p: p.name)


def main(argv: Optional[list[str]] = None) -> int:
    _semillas()
    _config_log()

    parser = argparse.ArgumentParser(
        description="Extracción de keyframes consciente del motion blur (Fase 1)."
    )
    parser.add_argument(
        "--carpeta-videos", type=Path, default=None, help="Carpeta de videos fuente."
    )
    parser.add_argument(
        "--video", type=Path, default=None, help="Ruta a un único video específico a procesar."
    )
    parser.add_argument(
        "--carpeta-salida", type=Path, default=None, help="Carpeta destino de imágenes."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=45,
        help="Máximo de frames por video (cupo total).",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Procesos paralelos (uno por video)."
    )
    args = parser.parse_args(argv)

    raiz = Path(__file__).resolve().parent.parent
    carpeta_videos: Path = args.carpeta_videos or raiz / "Videos para entrenamiento"
    carpeta_salida: Path = (
        args.carpeta_salida or raiz / "Dataset_YOLOv8" / "images_sin_etiquetar"
    )

    if args.video is not None:
        video_especifico = Path(args.video).resolve()
        if not video_especifico.is_file():
            LOGGER.error("Video especificado no existe: %s", video_especifico)
            return 2
        videos = [video_especifico]
    else:
        if not carpeta_videos.is_dir():
            LOGGER.error("Carpeta de videos no existe: %s", carpeta_videos)
            return 2
        videos = _descubrir_videos(carpeta_videos)

    cfg = ExtractConfig(
        max_frames_por_video=args.max_frames, workers=max(1, args.workers)
    )

    if not videos:
        LOGGER.error("No se encontraron videos a procesar.")
        return 2

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    _archivar_legado(carpeta_salida)

    LOGGER.info(
        "%d videos a procesar con %d workers (cupo %d frames/video).",
        len(videos),
        cfg.workers,
        cfg.max_frames_por_video,
    )

    t0 = time.perf_counter()
    resultados: list[VideoResult] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_procesar_video, v, carpeta_salida, cfg): v for v in videos
        }
        for fut in as_completed(futures):
            ruta = futures[fut]
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001 - reportar y continuar con el resto
                LOGGER.exception("Error procesando %s", ruta.name)
                continue
            resultados.append(res)
            LOGGER.info(
                "%s: %d frames | kine=%d blur=%d",
                res.nombre_video,
                res.total_frames,
                res.extraidos_kine,
                res.extraidos_blur,
            )

    resultados.sort(key=lambda r: r.nombre_video)
    total = sum(r.extraidos_kine + r.extraidos_blur for r in resultados)
    LOGGER.info(
        "Extracción completada: %d imágenes de %d videos en %.1f s.",
        total,
        len(resultados),
        time.perf_counter() - t0,
    )

    # --- Gate 1: verificación de integridad (sin imágenes corruptas de 0 bytes) ---
    corruptas = [p for p in carpeta_salida.glob("*.jpg") if p.stat().st_size == 0]
    if corruptas:
        LOGGER.error(
            "Gate 1 FALLIDO: %d imágenes de 0 bytes: %s",
            len(corruptas),
            [str(p) for p in corruptas[:5]],
        )
        return 1

    sin_prefijo = [
        p for p in carpeta_salida.glob("*.jpg") if not p.name.startswith("vid_")
    ]
    if sin_prefijo:
        LOGGER.warning(
            "Gate 1: %d imágenes sin prefijo video_id (verificar): %s",
            len(sin_prefijo),
            [p.name for p in sin_prefijo[:5]],
        )

    LOGGER.info(
        "Gate 1 OK: 0 imágenes corruptas; total=%d en %s", total, carpeta_salida
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
