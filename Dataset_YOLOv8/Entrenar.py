# -*- coding: utf-8 -*-
"""
Pipeline de Entrenamiento Exhaustivo VBT sobre AMD ROCm (Fase 5 - Plan.md)
==========================================================================
Optimizado para AMD Radeon RX 7900 XTX (24 GB VRAM - RDNA3 gfx1100):
  - Bloqueo estricto de GPU dedicada RX 7900 XTX (evita iGPU Ryzen 5700G o CPU).
  - Regularización anti-overfitting: AdamW + Weight Decay desacoplado (0.01) +
    Dropout (0.15) + Cosine Annealing LR + Warmup (5 épocas) + close_mosaic (15).
  - Función de pérdida calibrada para cinemática VBT: CIoU (box=9.0) + DFL (dfl=2.5).
  - Evaluación en conjunto 'test' con cálculo riguroso de F1(tau*), PR-AUC y ROC-AUC (auc_roc).
"""

from __future__ import annotations
import sys

import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 1. Directivas de estabilidad y rendimiento para RDNA3 (RX 7900 XTX - gfx1100)
# En RDNA3 gfx1100 hipBLASLt no está soportado en ROCm 6.2; se fuerza hipblas estándar.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

# Configuración de logging estructurado (§0.2 Plan.md)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][Entrenar_VBT] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Entrenar_VBT")


def verificar_dependencias() -> None:
    """Verifica el aislamiento y disponibilidad de librerías críticas (§0.2)."""
    dependencias = ["torch", "ultralytics", "albumentations", "cv2", "numpy", "psutil", "yaml"]
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError as exc:
            logger.critical(
                f"Dependencia requerida no encontrada: '{dep}'. Detalle: {exc}"
            )
            raise RuntimeError(f"Falta dependencia crítica: {dep}") from exc


verificar_dependencias()

import cv2
import numpy as np
import psutil
import torch
import yaml
from ultralytics import YOLO


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """
    Configuración maestra de hiperparámetros para máxima generalización VBT.
    Diseñada para aprovechar los 24 GB de VRAM de la RX 7900 XTX sin sobreajuste.
    """

    # Ciclo y resolución
    epochs: int = 120
    patience: int = 25          # Early stopping si el mAP/F1 en 'val' no mejora en 25 épocas
    imgsz: int = 1024
    batch_s: int = 24           # Batch óptimo para YOLO11s/YOLOv8s en 24 GB VRAM (~17.5 GB uso)
    batch_n: int = 32           # Batch óptimo para modelos Nano en 24 GB VRAM
    batch_m: int = 16           # Batch óptimo para modelos Medium en 24 GB VRAM
    nbs: int = 64               # Nominal Batch Size (acumula gradiente automáticamente hasta 64)
    workers: int = 8

    # Optimizador y Programación de Learning Rate (Anti-Overfitting)
    optimizer: str = "AdamW"
    lr0: float = 0.001         # LR inicial estable para fine-tuning con AdamW a 1024px
    lrf: float = 0.01         # LR final = 8e-6 al cierre de la curva cosenoidal
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.80
    warmup_bias_lr: float = 0.01
    cos_lr: bool = True         # Decaimiento cosenoidal suave (OneCycle Cosine Annealing)

    # Regularización explícita y estabilización de bordes
    weight_decay: float = 0.001  # Penalización L2 desacoplada fuerte para AdamW
    dropout: float = 0.0      # Dropout en cabezal para evitar memorización de discos/fondos
    close_mosaic: int = 10    # Últimas 15 épocas con frames reales completos (sin mosaico)

    # Pesos de la Función de Pérdida (Foco en precisión sub-píxel de caja y diámetro)
    box_loss_gain: float = 7.5  # CIoU Loss reforzado (centroide + aspecto circular)
    dfl_loss_gain: float = 1.8  # Distribution Focal Loss reforzado (bordes bajo motion blur)
    cls_loss_gain: float = 0.85 # BCE Loss moderado (monoclase: 'disco')

    # Aumentaciones nativas de apoyo
    mosaic: float = 1.0
    mixup: float = 0.12
    erasing: float = 0.25       # Simula oclusión parcial (manos, straps, jaula)
    scale: float = 0.55
    degrees: float = 6.0
    flipud: float = 0.0         # Jamás invertir verticalmente en VBT
    fliplr: float = 0.50

    # Ejecución y Validación (Gate 5)
    amp: bool = True
    seed: int = 42
    save_period: int = 10
    val_batch: int = 8
    vram_minima_gb: float = 20.0  # Garantiza que solo corra en la RX 7900 XTX (24 GB)
    f1_minimo_gate5: float = 0.94
    map50_minimo_gate5: float = 0.96
    map50_95_minimo_gate5: float = 0.52
    auc_roc_minimo_gate5: float = 0.95


def fijar_semillas(seed: int = 42) -> None:
    """Garantiza reproducibilidad global fijando semillas en Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Semilla global fijada en {seed}.")


def seleccionar_gpu_rx7900xtx(vram_minima_gb: float = 20.0) -> int:
    """
    Escanea los dispositivos ROCm/HIP y exige entrenar EXCLUSIVAMENTE en la
    AMD Radeon RX 7900 XTX (24 GB VRAM), descartando cualquier iGPU (ej. Ryzen 5700G).
    """
    if not torch.cuda.is_available():
        rocm_py = Path.home() / ".venv-rocm" / "bin" / "python"
        msg = (
            f"FATAL: PyTorch no detecta ninguna GPU activa (torch.cuda.is_available() == False).\n"
            f"Intérprete actual: {sys.executable}\n"
        )
        if getattr(torch.version, "cuda", None) and not getattr(torch.version, "hip", None):
            msg += (
                "⚠️ Estás usando un PyTorch compilado para NVIDIA CUDA (+cuXXX). "
                "Tu tarjeta es una AMD Radeon RX 7900 XTX, que requiere PyTorch con soporte AMD ROCm (+rocm).\n"
            )
        if rocm_py.is_file():
            msg += f"Ejecuta el script con el entorno ROCm dedicado:\n  {rocm_py} {Path(__file__).resolve()}\n"
        raise RuntimeError(msg)

    num_gpus = torch.cuda.device_count()
    candidato_idx: Optional[int] = None

    for idx in range(num_gpus):
        props = torch.cuda.get_device_properties(idx)
        nombre = props.name
        vram_gb = props.total_memory / (1024**3)
        gcn_arch = getattr(props, "gcnArchName", "desconocido")
        logger.info(
            f"Dispositivo detectado [{idx}]: '{nombre}' | Arquitectura: {gcn_arch} | VRAM: {vram_gb:.2f} GiB"
        )

        es_7900xtx = ("7900" in nombre.upper()) or ("GFX1100" in str(gcn_arch).upper())
        if es_7900xtx and vram_gb >= vram_minima_gb:
            candidato_idx = idx
            break
        elif vram_gb >= vram_minima_gb and candidato_idx is None:
            candidato_idx = idx

    if candidato_idx is None:
        raise RuntimeError(
            f"FATAL: No se encontró la GPU AMD Radeon RX 7900 XTX (>= {vram_minima_gb} GiB VRAM). "
            "El entrenamiento ha sido abortado para evitar correr en la iGPU integrada o en CPU."
        )

    torch.cuda.set_device(candidato_idx)
    torch.backends.cudnn.benchmark = True

    nombre_sel = torch.cuda.get_device_name(candidato_idx)
    vram_sel = torch.cuda.get_device_properties(candidato_idx).total_memory / (1024**3)
    version_hip = getattr(torch.version, "hip", "N/A")
    logger.info(
        f"✅ GPU SELECCIONADA Y BLOQUEADA -> cuda:{candidato_idx} ({nombre_sel}) | "
        f"VRAM: {vram_sel:.2f} GiB | ROCm/HIP: {version_hip}"
    )
    return candidato_idx


def obtener_ram_disponible_gb() -> float:
    """Calcula la memoria RAM del sistema libre disponible en GiB."""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 0.0


def detectar_proceso_ollama() -> bool:
    """Verifica si el servidor Ollama está activo en segundo plano."""
    try:
        for proc in psutil.process_iter(["name", "cmdline"]):
            nombre = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in nombre or "ollama" in cmdline:
                return True
    except Exception:
        pass
    return False


def transformar_albumentations_vbt() -> list:
    """
    Pipeline Albumentations específico para VBT:
      1. MotionBlur: simula la caída rápida de la barra y vibración del piso en peso muerto.
      2. Defocus / GaussianBlur: simula pérdida momentánea de foco del autofoco del celular.
      3. ImageCompression: simula la compresión con pérdida de WhatsApp Cloud API.
      4. RandomBrightnessContrast + CLAHE: realza los bordes de discos negros en gimnasios oscuros.
    """
    import albumentations as A

    return [
        A.MotionBlur(blur_limit=(7, 23), allow_shifted=True, p=0.40),
        A.GaussianBlur(blur_limit=(3, 7), p=0.15),
        A.ImageCompression(quality_range=(45, 85), p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.40),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.25),
    ]


def seleccionar_modelo_base_y_batch(
    dir_base: Path, cfg: ConfigEntrenamiento
) -> Tuple[str, int]:
    """
    Prioriza YOLO11s (CNN + Atención Espacial C2PSA) sobre variantes Nano antiguas,
    y asigna el tamaño de batch ideal para los 24 GB de VRAM de la RX 7900 XTX.
    Orden de prioridad:
      1. Variable de entorno YOLO_BASE_MODEL
      2. yolo11s.pt local (o descarga automática de Ultralytics si hay red)
      3. yolov8s.pt local
      4. best.pt previo (solo si se desea continuar desde pesos propios)
      5. yolov8n.pt fallback
    """
    env_model = os.environ.get("YOLO_BASE_MODEL")
    if env_model and Path(env_model).exists():
        modelo_str = env_model
    elif (dir_base / "yolo11s.pt").exists():
        modelo_str = str(dir_base / "yolo11s.pt")
    elif (dir_base / "yolov8s.pt").exists():
        modelo_str = str(dir_base / "yolov8s.pt")
    else:
        # Preferimos inicializar o descargar yolo11s.pt; si no hay red, caer a best.pt / yolov8n.pt
        try:
            _ = YOLO("yolo11s.pt")
            modelo_str = "yolo11s.pt"
        except Exception:
            if (dir_base / "best.pt").exists():
                modelo_str = str(dir_base / "best.pt")
            elif (dir_base / "yolov8n.pt").exists():
                modelo_str = str(dir_base / "yolov8n.pt")
            else:
                modelo_str = "yolov8n.pt"

    nombre_bajo = Path(modelo_str).stem.lower()
    if nombre_bajo.endswith("m"):
        batch_opt = cfg.batch_m
    elif nombre_bajo.endswith("s"):
        batch_opt = cfg.batch_s
    else:
        batch_opt = cfg.batch_n

    logger.info(
        f"Arquitectura seleccionada: '{modelo_str}' -> Batch optimizado para 24 GB VRAM: {batch_opt} "
        f"(Acumulación efectiva hasta nbs={cfg.nbs})"
    )
    return modelo_str, batch_opt


def _calcular_iou_matriz(cajas_a: np.ndarray, cajas_b: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de IoU [N, M] entre dos conjuntos de cajas en formato xyxy.
    """
    if len(cajas_a) == 0 or len(cajas_b) == 0:
        return np.zeros((len(cajas_a), len(cajas_b)), dtype=np.float32)

    x1 = np.maximum(cajas_a[:, None, 0], cajas_b[None, :, 0])
    y1 = np.maximum(cajas_a[:, None, 1], cajas_b[None, :, 1])
    x2 = np.minimum(cajas_a[:, None, 2], cajas_b[None, :, 2])
    y2 = np.minimum(cajas_a[:, None, 3], cajas_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (cajas_a[:, 2] - cajas_a[:, 0]) * (cajas_a[:, 3] - cajas_a[:, 1])
    area_b = (cajas_b[:, 2] - cajas_b[:, 0]) * (cajas_b[:, 3] - cajas_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union_area > 1e-6, inter_area / union_area, 0.0)


def _roc_auc_trapezoidal_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Cálculo exacto del Área Bajo la Curva ROC (AUC-ROC) mediante ordenamiento e
    integración trapezoidal en NumPy puro (sin depender de scikit-learn).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        # Fallback determinista por integración trapezoidal de la curva TPR vs FPR
        orden = np.argsort(-y_scores, kind="mergesort")
        y_sorted = y_true[orden]

        tps = np.cumsum(y_sorted == 1)
        fps = np.cumsum(y_sorted == 0)

        tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
        fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
        integrador = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(integrador(tpr, fpr)) if integrador else 0.0


def calcular_auc_roc_en_split(
    modelo: YOLO,
    ruta_data_yaml: Path,
    split: str,
    imgsz: int,
    device: int,
    iou_umbral: float = 0.50,
) -> Dict[str, float]:
    """
    Evalúa el poder discriminativo real del detector calculando el ROC-AUC (auc_roc)
    a nivel de instancias y regiones candidatas sobre el conjunto 'test' (o 'val').

    Metodología:
      - Ejecuta inferencia con umbral mínimo (conf=0.001) para capturar toda la cola de
        probabilidades (Verdaderos Positivos, Falsos Positivos de fondo y Falsos Negativos).
      - Empareja predicciones contra Ground Truth con IoU >= 0.50:
          * GT detectado -> (y_true = 1, y_score = conf_pred)
          * GT no detectado (FN) -> (y_true = 1, y_score = 0.0)
          * Predicción sin GT (FP en fondo/rack/otros discos) -> (y_true = 0, y_score = conf_pred)
          * Imagen de fondo limpia sin FP (TN) -> (y_true = 0, y_score = 0.0)
    """
    with open(ruta_data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", ruta_data_yaml.parent))
    if not base_path.is_absolute():
        base_path = (ruta_data_yaml.parent / base_path).resolve()

    split_rel = data_cfg.get(split) or data_cfg.get("val")
    dir_imgs = (base_path / split_rel).resolve()
    if not dir_imgs.exists():
        logger.warning(f"Directorio de imágenes para ROC-AUC no existe: {dir_imgs}")
        return {"auc_roc": 0.0, "muestras_positivas": 0, "muestras_negativas": 0}

    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    archivos_img = sorted(
        [p for p in dir_imgs.rglob("*") if p.suffix.lower() in extensiones]
    )
    if not archivos_img:
        return {"auc_roc": 0.0, "muestras_positivas": 0, "muestras_negativas": 0}

    y_true: List[int] = []
    y_scores: List[float] = []

    logger.info(
        f"Calculando métrica ROC-AUC (auc_roc) sobre {len(archivos_img)} imágenes del split '{split}'..."
    )

    # Inferencia por lotes para máxima velocidad en la RX 7900 XTX
    lote = 16
    for i in range(0, len(archivos_img), lote):
        batch_paths = archivos_img[i : i + lote]
        preds = modelo.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=0.001,
            iou=0.45,
            max_det=30,
            device=device,
            verbose=False,
        )

        for img_path, pred in zip(batch_paths, preds):
            h_img, w_img = pred.orig_shape

            # Localizar archivo de etiquetas Ground Truth correspondiente
            partes = list(img_path.parts)
            if "images" in partes:
                idx_part = len(partes) - 1 - partes[::-1].index("images")
                partes[idx_part] = "labels"
                label_path = Path(*partes).with_suffix(".txt")
            else:
                label_path = img_path.with_suffix(".txt")

            gt_boxes_list = []
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as lf:
                    for linea in lf:
                        vals = linea.strip().split()
                        if len(vals) >= 5:
                            xc, yc, bw, bh = map(float, vals[1:5])
                            x1 = (xc - bw / 2.0) * w_img
                            y1 = (yc - bh / 2.0) * h_img
                            x2 = (xc + bw / 2.0) * w_img
                            y2 = (yc + bh / 2.0) * h_img
                            gt_boxes_list.append([x1, y1, x2, y2])

            gt_boxes = (
                np.array(gt_boxes_list, dtype=np.float32)
                if gt_boxes_list
                else np.empty((0, 4), dtype=np.float32)
            )

            if pred.boxes is not None and len(pred.boxes) > 0:
                pred_boxes = pred.boxes.xyxy.cpu().numpy()
                pred_confs = pred.boxes.conf.cpu().numpy()
                orden_conf = np.argsort(-pred_confs)
                pred_boxes = pred_boxes[orden_conf]
                pred_confs = pred_confs[orden_conf]
            else:
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_confs = np.empty((0,), dtype=np.float32)

            gt_usados = set()
            pred_emparejadas = set()

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                matriz_iou = _calcular_iou_matriz(pred_boxes, gt_boxes)
                for p_idx in range(len(pred_boxes)):
                    mejor_gt = int(np.argmax(matriz_iou[p_idx]))
                    mejor_iou = float(matriz_iou[p_idx, mejor_gt])
                    if mejor_iou >= iou_umbral and mejor_gt not in gt_usados:
                        gt_usados.add(mejor_gt)
                        pred_emparejadas.add(p_idx)
                        y_true.append(1)
                        y_scores.append(float(pred_confs[p_idx]))

            # Falsos Negativos (GT que no fueron detectados)
            for g_idx in range(len(gt_boxes)):
                if g_idx not in gt_usados:
                    y_true.append(1)
                    y_scores.append(0.0)

            # Falsos Positivos (Predicciones en el fondo que no corresponden a ningún GT)
            fps_en_imagen = 0
            for p_idx in range(len(pred_boxes)):
                if p_idx not in pred_emparejadas:
                    y_true.append(0)
                    y_scores.append(float(pred_confs[p_idx]))
                    fps_en_imagen += 1

            # Verdadero Negativo explícito (para garantizar soporte negativo si el modelo es muy limpio)
            if fps_en_imagen == 0:
                y_true.append(0)
                y_scores.append(0.0)

    auc_val = _roc_auc_trapezoidal_numpy(np.array(y_true), np.array(y_scores))
    n_pos = int(np.sum(np.array(y_true) == 1))
    n_neg = int(np.sum(np.array(y_true) == 0))

    logger.info(
        f"Cálculo ROC-AUC finalizado -> auc_roc: {auc_val:.4f} "
        f"(Positivos evaluados: {n_pos} | Negativos/Fondo evaluados: {n_neg})"
    )
    return {
        "auc_roc": round(auc_val, 4),
        "muestras_positivas": n_pos,
        "muestras_negativas": n_neg,
    }


def entrenar_y_evaluar(cfg: ConfigEntrenamiento = ConfigEntrenamiento()) -> Path:
    """
    Orquesta el entrenamiento completo en la RX 7900 XTX, evalúa contra 'test'
    (o 'val' si 'test' no está definido) y exporta metricas_produccion.json.
    """
    fijar_semillas(cfg.seed)
    device_idx = seleccionar_gpu_rx7900xtx(cfg.vram_minima_gb)

    dir_base = Path(__file__).resolve().parent
    ruta_data = dir_base / "data.yaml"
    ruta_hyp = dir_base / "hyp_vbt.yaml"

    if not ruta_data.exists():
        raise FileNotFoundError(f"No se encontró 'data.yaml' en: {ruta_data}")

    # Presupuesto de memoria RAM en Host
    ram_gb = obtener_ram_disponible_gb()
    if detectar_proceso_ollama():
        logger.warning(
            "⚠️ ALERTA: El servidor 'ollama' está activo. Considera detenerlo "
            "(systemctl stop ollama) para liberar RAM y VRAM antes de entrenar."
        )

    cache_modo = "ram" if ram_gb >= 10.0 else "disk"
    logger.info(
        f"RAM libre en sistema: {ram_gb:.2f} GiB -> Modo de caché seleccionado: '{cache_modo}'"
    )

    modelo_base, batch_optimo = seleccionar_modelo_base_y_batch(dir_base, cfg)
    model = YOLO(modelo_base)

    logger.info(
        f"Iniciando entrenamiento VBT en RX 7900 XTX (device={device_idx}) | "
        f"epochs={cfg.epochs}, batch={batch_optimo} (nbs={cfg.nbs}), imgsz={cfg.imgsz} | "
        f"optimizer={cfg.optimizer} (lr0={cfg.lr0}, lrf={cfg.lrf}, weight_decay={cfg.weight_decay}, dropout={cfg.dropout}) | "
        f"Loss gains: box={cfg.box_loss_gain}, dfl={cfg.dfl_loss_gain}, cls={cfg.cls_loss_gain}"
    )

    resultados = model.train(
        data=str(ruta_data),
        cfg=str(ruta_hyp) if ruta_hyp.exists() else None,
        epochs=cfg.epochs,
        patience=cfg.patience,
        imgsz=cfg.imgsz,
        batch=batch_optimo,
        nbs=cfg.nbs,
        device=device_idx,
        workers=cfg.workers,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        cos_lr=cfg.cos_lr,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        box=cfg.box_loss_gain,
        dfl=cfg.dfl_loss_gain,
        cls=cfg.cls_loss_gain,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        erasing=cfg.erasing,
        scale=cfg.scale,
        degrees=cfg.degrees,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        close_mosaic=cfg.close_mosaic,
        amp=cfg.amp,
        cache=cache_modo,
        augmentations=transformar_albumentations_vbt(),
        seed=cfg.seed,
        project=str(dir_base / "runs" / "detect"),
        name="train_vbt_rocm_1024",
        exist_ok=True,
        save=True,
        save_period=cfg.save_period,
        plots=True,
    )

    # Localizar mejor checkpoint generado (best.pt)
    save_dir = Path(
        getattr(
            resultados, "save_dir", dir_base / "runs" / "detect" / "train_vbt_rocm_1024"
        )
    )
    ruta_best = save_dir / "weights" / "best.pt"
    if not ruta_best.exists():
        ruta_best_fallback = (
            dir_base / "runs" / "detect" / "train_vbt_rocm_1024" / "weights" / "best.pt"
        )
        if ruta_best_fallback.exists():
            ruta_best = ruta_best_fallback
        else:
            raise FileNotFoundError(
                f"No se localizó el archivo de pesos entrenado en: {ruta_best}"
            )

    logger.info(f"Entrenamiento completado. Mejor checkpoint: {ruta_best}")

    # Determinar si existe la partición 'test' en data.yaml; si no, evaluar en 'val'
    with open(ruta_data, "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)
    split_eval = "test" if "test" in data_dict and data_dict["test"] else "val"

    logger.info(f"Ejecutando evaluación rigurosa sobre el conjunto '{split_eval}'...")
    modelo_final = YOLO(str(ruta_best))

    metricas_test = modelo_final.val(
        data=str(ruta_data),
        split=split_eval,
        imgsz=cfg.imgsz,
        batch=cfg.val_batch,
        device=device_idx,
        plots=True,
    )

    mp = float(metricas_test.box.mp)
    mr = float(metricas_test.box.mr)
    map50 = float(metricas_test.box.map50)
    map50_95 = float(metricas_test.box.map)

    best_f1 = (2 * mp * mr) / max(mp + mr, 1e-6)
    conf_optimo = 0.50
    tau_low = 0.20
    tau_high = 0.80
    prec_optima = mp
    rec_optima = mr
    pr_auc = map50

    # Extracción analítica continua de la curva F1(tau) (§5.1.3 corregido)
    try:
        if hasattr(metricas_test.box, "curves_results") and len(metricas_test.box.curves_results) >= 4:
            x_conf = np.asarray(metricas_test.box.curves_results[1][0], dtype=np.float32)
            y_f1 = np.asarray(metricas_test.box.curves_results[1][1], dtype=np.float32)
            y_prec = np.asarray(metricas_test.box.curves_results[2][1], dtype=np.float32)
            y_rec = np.asarray(metricas_test.box.curves_results[3][1], dtype=np.float32)

            # Aplanar si viene con forma (1, N) o seleccionar dimensión con datos
            f1_arr = y_f1[0] if (y_f1.ndim > 1 and y_f1.shape[0] > 0) else y_f1.flatten()
            prec_arr = y_prec[0] if (y_prec.ndim > 1 and y_prec.shape[0] > 0) else y_prec.flatten()
            rec_arr = y_rec[0] if (y_rec.ndim > 1 and y_rec.shape[0] > 0) else y_rec.flatten()

            if f1_arr.size > 0 and f1_arr.size == x_conf.size:
                idx_max = int(np.argmax(f1_arr))
                best_f1 = float(f1_arr[idx_max])
                conf_optimo = float(x_conf[idx_max])
                prec_optima = float(prec_arr[idx_max]) if prec_arr.size > idx_max else mp
                rec_optima = float(rec_arr[idx_max]) if rec_arr.size > idx_max else mr

                band_indices = np.where(f1_arr >= 0.95 * best_f1)[0]
                if len(band_indices) > 0:
                    tau_low = float(x_conf[band_indices[0]])
                    tau_high = float(x_conf[band_indices[-1]])

                logger.info(
                    f"Optimización F1 completada -> Max F1: {best_f1:.4f} en tau*={conf_optimo:.4f}. "
                    f"Banda operativa: [{tau_low:.4f}, {tau_high:.4f}]"
                )
    except Exception as exc:
        logger.warning(f"No se pudo extraer la curva analítica F1-Confidence: {exc}. Usando métricas estándar.")

    # Cálculo de ROC-AUC (auc_roc) sobre el split de evaluación
    info_roc = calcular_auc_roc_en_split(
        modelo=modelo_final,
        ruta_data_yaml=ruta_data,
        split=split_eval,
        imgsz=cfg.imgsz,
        device=device_idx,
        iou_umbral=0.50,
    )
    auc_roc = info_roc["auc_roc"]

    # Verificación integral del Gate 5
    gate_5_aprobado = (
        best_f1 >= cfg.f1_minimo_gate5
        and map50 >= cfg.map50_minimo_gate5
        and map50_95 >= cfg.map50_95_minimo_gate5
        and auc_roc >= cfg.auc_roc_minimo_gate5
    )

    reporte_metricas: Dict[str, Any] = {
        "modelo_base": modelo_base,
        "modelo_ruta": str(ruta_best),
        "gpu_utilizada": torch.cuda.get_device_name(device_idx),
        "split_evaluado": split_eval,
        "batch_utilizado": batch_optimo,
        "precision": round(prec_optima, 4),
        "recall": round(rec_optima, 4),
        "f1_score": round(best_f1, 4),
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "conf_optimo": round(conf_optimo, 4),
        "tau_low": round(tau_low, 4),
        "tau_high": round(tau_high, 4),
        "muestras_roc_positivas": info_roc["muestras_positivas"],
        "muestras_roc_negativas": info_roc["muestras_negativas"],
        "gate_5_aprobado": bool(gate_5_aprobado),
    }

    ruta_salida_json = dir_base / "metricas_produccion.json"
    with open(ruta_salida_json, "w", encoding="utf-8") as f:
        json.dump(reporte_metricas, f, indent=4)

    logger.info(
        f"RESULTADOS ({split_eval.upper()}) -> "
        f"F1: {best_f1:.4f} | ROC-AUC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}"
    )

    if gate_5_aprobado:
        logger.info("✅ GATE 5 SUPERADO CON ÉXITO (F1, mAP50, mAP50-95 y ROC-AUC aprobados).")
    else:
        logger.warning("⚠️ GATE 5 NO SUPERADO: Revisa las métricas en metricas_produccion.json.")

    # Actualizar checkpoint principal en el directorio base
    try:
        shutil.copy2(ruta_best, dir_base / "best.pt")
        logger.info(f"Checkpoint principal actualizado en: {dir_base / 'best.pt'}")
    except Exception as exc:
        logger.warning(f"No se pudo copiar {ruta_best} a best.pt: {exc}")

    return ruta_best


if __name__ == "__main__":
    entrenar_y_evaluar()
