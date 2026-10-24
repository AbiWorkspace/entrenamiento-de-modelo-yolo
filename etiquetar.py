#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etiquetar.py — Lanzador directo para el Etiquetador Manual de Frames (YOLO VBT).

Uso rápido:
    python3 etiquetar.py                  # Abre imágenes sin etiquetar (images_sin_etiquetar)
    python3 etiquetar.py --cola           # Abre la cola de revisión humana (cola_revision_humana.txt)
    python3 etiquetar.py --sospechosas    # Abre etiquetas sospechosas (etiquetas_sospechosas.json)
    python3 etiquetar.py --dataset        # Revisa o edita imágenes ya en images/{train,val,test}
    python3 etiquetar.py --video <ruta>   # Carga y navega directamente un video MP4
    python3 etiquetar.py --auto-ia        # Activa pre-anotación asistida con YOLO por defecto
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Resolver raíz y agregar al path
RAIZ = Path(__file__).resolve().parent
ETIQUETADOR_SCRIPT = RAIZ / "Dataset_YOLOv8" / "etiquetador_rapido.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lanzador del Etiquetador Manual de Frames para YOLO VBT."
    )
    parser.add_argument(
        "--cola",
        action="store_true",
        help="Abrir directamente la cola de revisión humana (cola_revision_humana.txt).",
    )
    parser.add_argument(
        "--sospechosas",
        action="store_true",
        help="Abrir directamente las etiquetas sospechosas de auditoría.",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Abrir imágenes del dataset actual (train/val/test).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Ruta a un archivo MP4 para navegar y etiquetar frames directamente.",
    )
    parser.add_argument(
        "--auto-ia",
        action="store_true",
        help="Activar sugerencias asistidas de IA (YOLO) al cambiar de fotograma.",
    )
    args, extras = parser.parse_known_args()

    # Determinar modo
    if args.video:
        modo = "video"
    elif args.cola:
        modo = "cola"
    elif args.sospechosas:
        modo = "sospechosas"
    elif args.dataset:
        modo = "dataset"
    else:
        modo = "sin_etiquetar"

    cmd_args = ["--mode", modo]
    if args.video:
        cmd_args.extend(["--video", str(args.video.resolve())])
    if args.auto_ia:
        cmd_args.append("--auto-suggest")
    cmd_args.extend(extras)

    # Importar y ejecutar directamente el módulo
    sys.path.insert(0, str(RAIZ / "Dataset_YOLOv8"))
    import etiquetador_rapido
    return etiquetador_rapido.main(cmd_args)


if __name__ == "__main__":
    sys.exit(main())
