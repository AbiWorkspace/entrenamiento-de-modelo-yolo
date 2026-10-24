from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - entorno sin torch
    raise SystemExit(f"No se pudo importar torch: {exc}") from exc
try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - entorno sin ultralytics
    raise SystemExit(f"No se pudo importar ultralytics: {exc}") from exc

DIRECTORIO_ACTUAL: Path = Path(__file__).resolve().parent
LOG: logging.Logger = logging.getLogger("auto_etiquetar")


@dataclass(frozen=True)
class ConfigEtiquetado:
    """Parámetros de inferencia y enrutamiento del auto-etiquetado."""

    input_dir: Path = DIRECTORIO_ACTUAL / "images_sin_etiquetar"
    output_img_dir: Path = DIRECTORIO_ACTUAL / "images" / "train"
    output_label_dir: Path = DIRECTORIO_ACTUAL / "labels" / "train"
    cola_revision: Path = DIRECTORIO_ACTUAL / "cola_revision_humana.txt"
    modelo: Path = DIRECTORIO_ACTUAL / "best.pt"
    imgsz: int = 1024
    conf_low: float = 0.15
    conf_high: float = 0.35
    iou: float = 0.45
    ar_min_aceptado: float = 0.35
    ar_max_aceptado: float = 1.65
    yc_zona_suelo: float = 0.68
    batch_size: int = 16
    seed: int = 42
    max_background_samples: int = 45
    overwrite: bool = True


def _seleccionar_dispositivo() -> str:
    if torch.cuda.is_available():
        LOG.info("Dispositivo de inferencia: GPU (%s)", torch.cuda.get_device_name(0))
        return "0"
    LOG.warning("CUDA no disponible; usando CPU.")
    return "cpu"


def _es_disco_aceptable(
    conf: float, w_px: float, h_px: float, cfg: ConfigEtiquetado
) -> bool:
    """Criterio de aceptación: confianza alta + relación de aspecto plausible."""
    if conf < cfg.conf_high:
        return False
    if h_px <= 0:
        return False
    ar = w_px / h_px
    return cfg.ar_min_aceptado <= ar <= cfg.ar_max_aceptado


def _etiquetar_lote(
    rutas: list[Path],
    model: YOLO,
    device: str,
    cfg: ConfigEtiquetado,
) -> tuple[list[str], list[Path], int]:
    """Infiere un lote y aplica el enrutamiento inteligente.

    Returns:
        (lineas_yaml_por_imagen, imagenes_a_revisar, total_cajas_aceptadas)
    """
    resultados = model.predict(
        source=[str(p) for p in rutas],
        conf=cfg.conf_low,
        iou=cfg.iou,
        augment=False,
        imgsz=cfg.imgsz,
        device=device,
        verbose=False,
    )

    lineas_por_imagen: list[str] = []
    a_revisar: list[Path] = []
    total_aceptadas: int = 0

    for ruta, res in zip(rutas, resultados):
        lineas: list[str] = []
        requiere_revision = False

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            img_h, img_w = res.orig_shape
            xywhn = np.asarray(boxes.xywhn.cpu().numpy())
            confs = np.asarray(boxes.conf.cpu().numpy())
            clases = np.asarray(boxes.cls.cpu().numpy())

            for (xc, yc, w, h), conf, cls_id in zip(xywhn, confs, clases):
                aceptado = _es_disco_aceptable(
                    float(conf), float(w) * img_w, float(h) * img_h, cfg
                )
                if aceptado:
                    lineas.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    total_aceptadas += 1
                # Revisión humana: confianza media, AR dudoso, o Zona Suelo (yc >= 0.68)
                if (
                    not aceptado
                    or float(yc) >= cfg.yc_zona_suelo
                    or float(conf) < cfg.conf_high
                ):
                    requiere_revision = True

        lineas_por_imagen.append("".join(lineas))
        if requiere_revision or not lineas:
            a_revisar.append(ruta)

    return lineas_por_imagen, a_revisar, total_aceptadas


def ejecutar_auto_etiquetado(cfg: ConfigEtiquetado) -> int:
    """Orquesta el auto-etiquetado sobre ``cfg.input_dir`` con control estricto de fondos."""
    import re
    from collections import defaultdict

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if not cfg.modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {cfg.modelo}")
    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Directorio de entrada inexistente: {cfg.input_dir}")

    extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imagenes = sorted(
        p
        for p in cfg.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensiones
    )
    LOG.info("Imágenes en %s: %d", cfg.input_dir, len(imagenes))

    cfg.output_img_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(cfg.modelo))
    device = _seleccionar_dispositivo()

    cola_total: list[Path] = []
    cajas_aceptadas = 0

    # Inferencia en lotes
    resultados_dict: dict[Path, str] = {}
    for inicio in range(0, len(imagenes), cfg.batch_size):
        lote = imagenes[inicio : inicio + cfg.batch_size]
        lineas_por_imagen, a_revisar, n_aceptadas = _etiquetar_lote(
            lote, model, device, cfg
        )
        cola_total.extend(a_revisar)
        cajas_aceptadas += n_aceptadas
        for ruta, contenido in zip(lote, lineas_por_imagen):
            resultados_dict[ruta] = contenido

        procesadas = min(inicio + cfg.batch_size, len(imagenes))
        LOG.info("[%d/%d] lotes inferidos", procesadas, len(imagenes))

    # Agrupar por video_id para clasificar videos con discos vs videos de fondo puro
    vid_re = re.compile(r"^(vid_[^_]+_[^_]+)")
    por_video: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for ruta, contenido in resultados_dict.items():
        m = vid_re.match(ruta.name)
        vid_id = m.group(1) if m else "vid_misc"
        por_video[vid_id].append((ruta, contenido))

    videos_con_disco = {
        vid for vid, items in por_video.items() if any(c.strip() for _, c in items)
    }
    videos_sin_disco = set(por_video.keys()) - videos_con_disco

    LOG.info("Videos detectados con discos: %d", len(videos_con_disco))
    LOG.info("Videos detectados como fondo puro: %d", len(videos_sin_disco))

    guardadas_disco = 0
    guardadas_fondo = 0

    # 1. Guardar frames con discos de los videos de levantamiento
    for vid in videos_con_disco:
        for ruta, contenido in por_video[vid]:
            if contenido.strip():
                dest_img = cfg.output_img_dir / ruta.name
                dest_txt = cfg.output_label_dir / f"{ruta.stem}.txt"
                dest_img.write_bytes(ruta.read_bytes())
                dest_txt.write_text(contenido, encoding="utf-8")
                guardadas_disco += 1

    # 2. Guardar muestra estratificada de fondos puros (hasta max_background_samples)
    cuota_por_video = max(1, cfg.max_background_samples // max(1, len(videos_sin_disco)))
    for vid in sorted(videos_sin_disco):
        frames_fondo = por_video[vid]
        # Muestreo espaciado y determinista dentro de cada video de fondo
        paso = max(1, len(frames_fondo) // cuota_por_video)
        seleccionados = frames_fondo[::paso][:cuota_por_video]
        for ruta, _ in seleccionados:
            if guardadas_fondo >= cfg.max_background_samples:
                break
            dest_img = cfg.output_img_dir / ruta.name
            dest_txt = cfg.output_label_dir / f"{ruta.stem}.txt"
            dest_img.write_bytes(ruta.read_bytes())
            dest_txt.write_text("", encoding="utf-8")  # Hard negative legítimo
            guardadas_fondo += 1

    # Cola de revisión humana (apendemos de forma idempotente)
    existentes: set[str] = set()
    if cfg.cola_revision.exists():
        existentes = {
            linea.strip()
            for linea in cfg.cola_revision.read_text(encoding="utf-8").splitlines()
        }
    nuevas = [str(p) for p in cola_total if str(p) not in existentes]
    if nuevas:
        with cfg.cola_revision.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(nuevas) + "\n")

    pct_fondo = (
        (guardadas_fondo / (guardadas_disco + guardadas_fondo) * 100)
        if (guardadas_disco + guardadas_fondo) > 0
        else 0
    )
    LOG.info("=" * 60)
    LOG.info("Auto-etiquetado finalizado.")
    LOG.info("  Imágenes con disco guardadas:  %d", guardadas_disco)
    LOG.info("  Imágenes de fondo puro:        %d (%.1f%% del dataset)", guardadas_fondo, pct_fondo)
    LOG.info("  Total dataset generado:        %d", guardadas_disco + guardadas_fondo)
    LOG.info("  Cajas aceptadas:               %d", cajas_aceptadas)
    LOG.info("  Salida imágenes:               %s", cfg.output_img_dir)
    LOG.info("  Salida labels:                 %s", cfg.output_label_dir)
    return 0


def _construir_config(argv: list[str]) -> ConfigEtiquetado:
    conf_high: float = 0.35
    conf_low: float = 0.15
    max_bg: int = 45
    overwrite: bool = True
    input_dir: Path = DIRECTORIO_ACTUAL / "images_sin_etiquetar"
    it = iter(argv)
    for arg in it:
        if arg == "--conf-high":
            conf_high = float(next(it))
        elif arg == "--conf-low":
            conf_low = float(next(it))
        elif arg == "--max-bg":
            max_bg = int(next(it))
        elif arg == "--no-overwrite":
            overwrite = False
        elif arg == "--input-dir":
            input_dir = Path(next(it)).resolve()
    return ConfigEtiquetado(
        conf_high=conf_high,
        conf_low=conf_low,
        max_background_samples=max_bg,
        overwrite=overwrite,
        input_dir=input_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.exit(ejecutar_auto_etiquetado(_construir_config(sys.argv[1:])))
