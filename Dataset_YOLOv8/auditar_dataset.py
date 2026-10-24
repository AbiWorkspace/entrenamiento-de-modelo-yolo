"""Auditoría de saneamiento del dataset YOLO (Fase 2).

Recorre todos los .txt bajo Dataset_YOLOv8/, valida class_id, recorta
coordenadas fuera de [0,1], detecta cajas diminutas o con relación de
aspecto inusual y emite un histograma de zonas verticales.

Uso:
    python auditar_dataset.py --fix-classes --report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 es opcional (solo para aspecto px)
    cv2 = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("auditar_dataset")

DIRECTORIO_BASE: Path = Path(__file__).resolve().parent
ARCHIVO_CLASSES: Path = DIRECTORIO_BASE / "classes.txt"
REPORTE_SOSPECHOSAS: Path = DIRECTORIO_BASE / "etiquetas_sospechosas.json"

AREA_MINIMA_NORM: float = 0.0008
ASPECTO_PX_MIN: float = 0.65
ASPECTO_PX_MAX: float = 1.45
YC_SUELO: float = 0.68
YC_ALTA: float = 0.38


@dataclass(frozen=True)
class ConfigAuditoria:
    """Parámetros de la auditoría del dataset."""

    directorio_base: Path = DIRECTORIO_BASE
    fix_classes: bool = False
    emitir_reporte: bool = False
    area_minima_norm: float = AREA_MINIMA_NORM
    aspecto_px_min: float = ASPECTO_PX_MIN
    aspecto_px_max: float = ASPECTO_PX_MAX


@dataclass
class EstadisticasLinea:
    """Acumuladores globales de la auditoría (mutable a propósito)."""

    total_lineas: int = 0
    errores_class_id: int = 0
    lineas_cortadas: int = 0
    cajas_sospechosas: int = 0
    zona_suelo: int = 0
    zona_media: int = 0
    zona_alta: int = 0


def _dimensiones_imagen(ruta_imagen: Path | None) -> tuple[int, int] | None:
    """Devuelve (ancho, alto) de la imagen o None si no se puede leer."""

    if cv2 is None or ruta_imagen is None or not ruta_imagen.exists():
        return None
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        return None
    alto, ancho = img.shape[:2]
    return (ancho // 8 * 8, alto // 8 * 8)  # reducida por 8x; escala proporcional


def _auditar_archivo(
    ruta_txt: Path,
    cfg: ConfigAuditoria,
    stats: EstadisticasLinea,
    sospechosas: list[dict[str, Any]],
) -> None:
    """Audita un único archivo de etiquetas YOLO y acumula en `stats`.

    Si `fix_classes` está activo, reescribe el archivo con class_id=0 y
    coordenadas recortadas. Idempotente: archivos ya saneados no cambian.
    """

    lineas_brutas: list[str] = ruta_txt.read_text(encoding="utf-8").splitlines()
    dimensiones = _dimensiones_imagen(ruta_txt.with_suffix(".jpg"))

    for linea in lineas_brutas:
        linea = linea.strip()
        if not linea:
            continue
        tokens: list[str] = linea.split()
        class_id: int = int(tokens[0])
        coords: tuple[float, float, float, float] = (
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            float(tokens[4]),
        )
        xc, yc, w, h = coords

        stats.total_lineas += 1
        if class_id != 0:
            stats.errores_class_id += 1

        clip: np.ndarray = np.clip(np.asarray(coords, dtype=np.float64), 0.0, 1.0)
        xc_c, yc_c, w_c, h_c = (float(v) for v in clip)
        if not np.allclose(np.asarray(coords, dtype=np.float64), clip):
            stats.lineas_cortadas += 1

        area_norm: float = float(w_c * h_c)
        motivo: list[str] = []
        if class_id != 0:
            motivo.append(f"class_id={class_id}")
        if area_norm < cfg.area_minima_norm:
            motivo.append(f"area_norm={area_norm:.6f} < {cfg.area_minima_norm}")
        if dimensiones is not None:
            w_px, h_px = w_c * dimensiones[0], h_c * dimensiones[1]
            aspecto = w_px / h_px if h_px > 0 else float("inf")
            if not (cfg.aspecto_px_min <= aspecto <= cfg.aspecto_px_max):
                motivo.append(
                    f"aspecto_px={aspecto:.2f} fuera de [{cfg.aspecto_px_min}, {cfg.aspecto_px_max}]"
                )

        if motivo:
            stats.cajas_sospechosas += 1
            sospechosas.append(
                {
                    "archivo": str(ruta_txt.relative_to(cfg.directorio_base)),
                    "motivos": motivo,
                    "valores": {
                        "class_id": class_id,
                        "xc": xc_c,
                        "yc": yc_c,
                        "w": w_c,
                        "h": h_c,
                    },
                }
            )

        if yc_c >= YC_SUELO:
            stats.zona_suelo += 1
        elif yc_c >= YC_ALTA:
            stats.zona_media += 1
        else:
            stats.zona_alta += 1

    if cfg.fix_classes:
        nuevas_lineas: list[str] = []
        for linea in lineas_brutas:
            linea_s = linea.strip()
            if not linea_s:
                continue
            tokens = linea_s.split()
            clipped = (float(np.clip(float(x), 0.0, 1.0)) for x in tokens[1:5])
            nuevas_lineas.append("0 " + " ".join(f"{v:.6f}" for v in clipped))
        ruta_txt.write_text(
            "\n".join(nuevas_lineas) + ("\n" if nuevas_lineas else ""), encoding="utf-8"
        )


def _verificar_classes(cfg: ConfigAuditoria) -> bool:
    """Verifica que classes.txt contenga única y exclusivamente 'disco'."""

    if not ARCHIVO_CLASSES.exists():
        logger.error("classes.txt no existe en %s", DIRECTORIO_BASE)
        return False
    lineas: list[str] = [
        ln.strip()
        for ln in ARCHIVO_CLASSES.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    ok: bool = lineas == ["disco"]
    if ok:
        logger.info("classes.txt OK: %s", lineas)
    else:
        logger.warning("classes.txt contiene: %s (esperado ['disco'])", lineas)
    return ok


def _historial_zonas(stats: EstadisticasLinea) -> None:
    """Imprime el histograma de distribución por zona vertical."""

    total_instancias: int = stats.zona_suelo + stats.zona_media + stats.zona_alta
    if total_instancias == 0:
        logger.warning("No se encontraron instancias etiquetadas.")
        return
    print("\n" + "=" * 52)
    print("HISTOGRAMA DE ZONAS VERTICALES (yc)")
    print("=" * 52)
    for etiqueta, conteo in (
        ("Zona Suelo  (yc >= 0.68)", stats.zona_suelo),
        ("Zona Media  (0.38 <= yc < 0.68)", stats.zona_media),
        ("Zona Alta   (yc < 0.38)", stats.zona_alta),
    ):
        pct: float = 100.0 * conteo / total_instancias
        barra: str = "#" * int(pct // 2)
        print(f"  {etiqueta}: {conteo:>6} ({pct:5.1f}%) {barra}")
    print("=" * 52)


def ejecutar_auditoria(cfg: ConfigAuditoria) -> int:
    """Ejecuta la auditoría completa y retorna el código de salida (Gate 2)."""

    stats = EstadisticasLinea()
    sospechosas: list[dict[str, Any]] = []

    _verificar_classes(cfg)

    excluidos = {"classes.txt", "cola_revision_humana.txt"}
    archivos_txt: list[Path] = sorted(
        f for f in cfg.directorio_base.rglob("*.txt") if f.name not in excluidos
    )
    logger.info("Archivos .txt a auditar: %d", len(archivos_txt))

    for ruta_txt in archivos_txt:
        _auditar_archivo(ruta_txt, cfg, stats, sospechosas)

    logger.info("Líneas analizadas: %d", stats.total_lineas)
    logger.info("  - Errores class_id != 0: %d", stats.errores_class_id)
    logger.info(
        "  - Líneas recortadas (coord fuera de [0,1]): %d", stats.lineas_cortadas
    )
    logger.info("  - Cajas sospechosas (área/aspecto): %d", stats.cajas_sospechosas)

    _historial_zonas(stats)

    if cfg.emitir_reporte or sospechosas:
        payload: dict[str, Any] = {
            "total_lineas": stats.total_lineas,
            "errores_class_id": stats.errores_class_id,
            "lineas_cortadas": stats.lineas_cortadas,
            "cajas_sospechosas": stats.cajas_sospechosas,
            "zonas": {
                "suelo_yc>=0.68": stats.zona_suelo,
                "media_0.38-0.68": stats.zona_media,
                "alta_yc<0.38": stats.zona_alta,
            },
            "sospechosas": sospechosas,
        }
        REPORTE_SOSPECHOSAS.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Reporte escrito en: %s", REPORTE_SOSPECHOSAS)

    # ---- Gate 2 ----
    total_instancias = stats.zona_suelo + stats.zona_media + stats.zona_alta
    pct_suelo: float = (
        (100.0 * stats.zona_suelo / total_instancias) if total_instancias else 0.0
    )
    gate_ok: bool = (
        stats.errores_class_id == 0 and stats.lineas_cortadas == 0 and pct_suelo >= 25.0
    )
    print(
        f"\nGATE 2: class_errors={stats.errores_class_id}, OOR={stats.lineas_cortadas}, "
        f"suelo={pct_suelo:.1f}% (>=25%) -> {'APROBADO' if gate_ok else 'RECHAZADO'}"
    )
    return 0 if gate_ok else 1


def _parse_args(argv: list[str] | None = None) -> ConfigAuditoria:
    parser = argparse.ArgumentParser(
        description="Auditar y sanea r el dataset YOLO (Fase 2)."
    )
    parser.add_argument(
        "--fix-classes",
        action="store_true",
        help="Corrige class_id a 0 y recorta coordenadas.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Escribe etiquetas_sospechosas.json aunque no haya sospechosas.",
    )
    args = parser.parse_args(argv)
    return ConfigAuditoria(fix_classes=args.fix_classes, emitir_reporte=args.report)


if __name__ == "__main__":
    sys.exit(ejecutar_auditoria(_parse_args()))
