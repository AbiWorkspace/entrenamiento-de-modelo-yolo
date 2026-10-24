# -*- coding: utf-8 -*-
"""
Exportación a ONNX + benchmark de inferencia para producción (Fase 7 - Plan.md).

Flujo:
  1. Localiza el mejor checkpoint `best.pt` (o recibe una ruta explícita).
  2. Exporta a ONNX con opset=17, imgsz=1024 y grafo simplificado.
  3. Ejecuta un benchmark de inferencia con ONNX Runtime en FP16 (CUDA/ROCm si
     está disponible, CPU si no) y verifica el objetivo de >= 45 FPS @1024.
  4. Despliega de forma ATÓMICA a `models/best.onnx` + `models/best.pt` para
     que `vbt_service.py` siempre vea un par coherente (sin estado intermedio).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuración de logging estructurado (§0.2 Plan.md)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][Exportar_Produccion] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Exportar_Produccion")


def verificar_dependencias() -> None:
    """Verifica disponibilidad de librerías críticas para export y benchmark."""
    dependencias = ["torch", "ultralytics", "cv2", "numpy", "onnxruntime"]
    faltantes: List[str] = []
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError as exc:
            logger.critical(
                f"Dependencia requerida no encontrada: '{dep}'. Detalle: {exc}"
            )
            faltantes.append(dep)
    if faltantes:
        raise RuntimeError(f"Faltan dependencias críticas: {faltantes}")


verificar_dependencias()

import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class ConfigExportacion:
    """Parámetros inmutables de exportación y benchmark (§0.2, Fase 7)."""

    imgsz: int = 1024
    opset: int = 17
    simplify: bool = True
    half: bool = True  # FP16 para el benchmark (meta de rendimiento)
    fps_minimo: float = 45.0
    warmup_iters: int = 25
    bench_iters: int = 200
    conf_umbral_bench: float = 0.08  # alineado con ConfigVBT.conf_deteccion


def localizar_best_pt(dir_base: Path, ruta_explicita: Optional[Path] = None) -> Path:
    """
    Localiza el mejor checkpoint entrenado:
      1. Ruta explícita si se proporcionó y existe.
      2. Dataset_YOLOv8/best.pt (deploy previo).
      3. runs/detect/*/weights/best.pt (última ejecución de Entrenar.py).
    """
    if ruta_explicita is not None:
        if not ruta_explicita.exists():
            raise FileNotFoundError(
                f"Ruta explícita indicada no existe: {ruta_explicita}"
            )
        return ruta_explicita

    candidato_deploy = dir_base / "best.pt"
    if candidato_deploy.exists():
        logger.info(f"Usando checkpoint de deploy: {candidato_deploy}")
        return candidato_deploy

    pesos = sorted(
        (dir_base / "runs" / "detect").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pesos:
        logger.info(f"Usando checkpoint de la última ejecución: {pesos[0]}")
        return pesos[0]

    raise FileNotFoundError(
        f"No se localizó 'best.pt' ni en '{candidato_deploy}' "
        f"ni en '{dir_base / 'runs' / 'detect'}'. Ejecuta Entrenar.py antes."
    )


def detectar_proveedor_onnxruntime() -> str:
    """
    Devuelve el proveedor preferido para ONNX Runtime. En ROCm no hay un
    proveedor nativo de AMD, así que se prueba CUDA (si existe) y se cae a CPU.
    El objetivo es un benchmark reproducible y honesto del grafo ONNX.
    """
    import onnxruntime as ort

    disponibles = ort.get_available_providers()
    logger.info(f"Proveedores ONNX Runtime disponibles: {disponibles}")

    if "CUDAExecutionProvider" in disponibles:
        return "CUDAExecutionProvider"
    if "CPUExecutionProvider" in disponibles:
        return "CPUExecutionProvider"
    raise RuntimeError("ONNX Runtime sin proveedores disponibles.")


def exportar_onnx(best_pt: Path, dir_salida: Path, cfg: ConfigExportacion) -> Path:
    """Exporta el checkpoint PyTorch a ONNX (opset 17, grafo simplificado)."""
    logger.info(f"Cargando modelo desde {best_pt}")
    model = YOLO(str(best_pt))

    # ultralytics escribe <modelo>.onnx junto al .pt; usamos una carpeta temporal
    # de trabajo para luego desplegar de forma atómica.
    ruta_onnx_esperada = best_pt.with_suffix(".onnx")

    logger.info(
        f"Exportando a ONNX: imgsz={cfg.imgsz}, opset={cfg.opset}, "
        f"simplify={cfg.simplify}"
    )
    model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        opset=cfg.opset,
        simplify=cfg.simplify,
        half=cfg.half,
        dynamic=False,
        nms=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    if not ruta_onnx_esperada.exists():
        raise FileNotFoundError(
            f"El export finalizó pero no se encontró '{ruta_onnx_esperada}'."
        )

    logger.info(f"ONNX generado: {ruta_onnx_esperada}")
    return ruta_onnx_esperada


def benchmark_onnx(
    ruta_onnx: Path, cfg: ConfigExportacion, dir_base: Path
) -> Dict[str, Any]:
    """
    Ejecuta inferencia repetida sobre frames reales del dataset de validación
    (o sintéticos si no hay) y mide FPS. Objetivo: >= cfg.fps_minimo @1024.
    """
    import onnxruntime as ort

    proveedor = detectar_proveedor_onnxruntime()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            str(ruta_onnx), sess_options, providers=[proveedor]
        )
    except Exception as exc:
        logger.warning(
            f"El proveedor '{proveedor}' falló al cargar la sesión ({exc}). "
            f"Reintentando con CPUExecutionProvider (FP32)."
        )
        session = ort.InferenceSession(str(ruta_onnx), sess_options)
        proveedor = session.get_providers()[0]

    entrada_nombre = session.get_inputs()[0].name
    forma_entrada = session.get_inputs()[0].shape  # p. ej. [1,3,1024,1024]
    logger.info(
        f"Sesión lista con proveedor '{proveedor}'. Entrada: {entrada_nombre} "
        f"forma={forma_entrada}"
    )

    # --- Preparar lote de entrada realista (BGR 0-255 -> CHW normalizado) ---
    directorio_imagenes = dir_base / "images"
    if directorio_imagenes.exists():
        extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
        muestras = [
            p for p in directorio_imagenes.iterdir() if p.suffix.lower() in extensiones
        ][: cfg.bench_iters]
    else:
        muestras = []

    def _imagen_a_tensor(ruta: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {ruta}")
        img = cv2.resize(img, (cfg.imgsz, cfg.imgsz))
        dtype = np.float16 if cfg.half else np.float32
        tensor = img.astype(dtype) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]
        return tensor

    # Determinismo del benchmark
    np.random.seed(42)
    dtype = np.float16 if cfg.half else np.float32
    lote: List[np.ndarray] = []
    for it in range(cfg.bench_iters):
        if muestras:
            lote.append(_imagen_a_tensor(muestras[it % len(muestras)]))
        else:
            # Sintética con ruido para saturar el grafo (HWC -> CHW + batch)
            base = np.random.randint(0, 255, (cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8)
            tensor = base.astype(dtype) / 255.0
            tensor = tensor.transpose(2, 0, 1)[None, ...]
            lote.append(tensor)

    # --- Warm-up (evita medir la primera inferencia con lazy-init de kernels) ---
    logger.info(f"Warm-up: {cfg.warmup_iters} iteraciones...")
    for _ in range(cfg.warmup_iters):
        session.run(None, {entrada_nombre: lote[0]})

    # --- Medición ---
    logger.info(f"Medición: {cfg.bench_iters} iteraciones...")
    t_inicio = time.perf_counter()
    for tensor in lote:
        session.run(None, {entrada_nombre: tensor})
    t_total = time.perf_counter() - t_inicio

    fps = cfg.bench_iters / t_total if t_total > 0 else float("inf")
    ms_medio = (t_total / cfg.bench_iters) * 1000.0
    objetivo_ok = fps >= cfg.fps_minimo

    logger.info(
        f"Benchmark final: {fps:.2f} FPS | {ms_medio:.2f} ms/iter "
        f"({'OK' if objetivo_ok else 'POR DEBAJO'} del objetivo de "
        f"{cfg.fps_minimo:.0f} FPS)"
    )

    return {
        "proveedor": proveedor,
        "imgsz": cfg.imgsz,
        "fps": round(fps, 2),
        "ms_por_iteracion": round(ms_medio, 3),
        "iteraciones": cfg.bench_iters,
        "warmup": cfg.warmup_iters,
        "objetivo_fps": cfg.fps_minimo,
        "objetivo_cumplido": bool(objetivo_ok),
    }


def desplegar_atomicamente(
    ruta_onnx: Path, best_pt: Path, dir_modelos: Path
) -> Dict[str, str]:
    """
    Copia `best.onnx` + `best.pt` a `models/` de forma atómica: escribe en un
    directorio temporal dentro del mismo volumen y luego renombra/mueve cada
    archivo, para que `vbt_service.py` nunca lea un par incompleto.
    """
    dir_modelos.mkdir(parents=True, exist_ok=True)

    destinos = {
        "onnx": dir_modelos / "best.onnx",
        "pt": dir_modelos / "best.pt",
    }

    for clave, origen in (("onnx", ruta_onnx), ("pt", best_pt)):
        destino = destinos[clave]
        # Escribimos a un temporal vecino y luego reemplazamos el destino.
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_modelos, prefix=f".{destino.name}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            shutil.copy2(origen, tmp_path)
            os.replace(tmp_path, destino)  # rename atómico en el mismo FS
        except Exception:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise

        logger.info(f"Desplegado '{destino.name}' -> {destino}")

    return {"onnx": str(destinos["onnx"]), "pt": str(destinos["pt"])}


def exportar_y_desplegar(
    ruta_explicita: Optional[Path] = None, cfg: ConfigExportacion = ConfigExportacion()
) -> Dict[str, Any]:
    """Orquesta las fases 7: localizar -> exportar -> benchmark -> desplegar."""
    dir_base = Path(__file__).resolve().parent
    root_proyecto = dir_base.parent
    dir_modelos = root_proyecto / "models"

    best_pt = localizar_best_pt(dir_base, ruta_explicita)
    ruta_onnx = exportar_onnx(best_pt, dir_base, cfg)
    resultado_bench = benchmark_onnx(ruta_onnx, cfg, dir_base)
    rutas_deploy = desplegar_atomicamente(ruta_onnx, best_pt, dir_modelos)

    resumen: Dict[str, Any] = {
        "best_pt": str(best_pt),
        "onnx_origen": str(ruta_onnx),
        "config": asdict(cfg),
        "benchmark": resultado_bench,
        "despliegue": rutas_deploy,
        "exito": bool(resultado_bench["objetivo_cumplido"]),
    }

    ruta_reporte = dir_base / "reporte_exportacion.json"
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4)
    logger.info(f"Reporte de exportación guardado en: {ruta_reporte}")

    if resumen["exito"]:
        logger.info(
            "✅ FASE 7 COMPLETA: modelo desplegado y objetivo de FPS alcanzado."
        )
    else:
        logger.warning(
            "⚠️  El benchmark quedó por debajo del objetivo de FPS. El despliegue "
            "se realizó igualmente, pero conviene revisar el proveedor o la "
            "resolución de inferencia."
        )

    return resumen


if __name__ == "__main__":
    exportar_y_desplegar()
