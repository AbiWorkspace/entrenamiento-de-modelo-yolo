# -*- coding: utf-8 -*-
"""
test_etiquetado_manual.py — Pruebas unitarias de la infraestructura de etiquetado manual.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF

import sys
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
RAIZ = DIRECTORIO_ACTUAL.parent
sys.path.insert(0, str(RAIZ / "Dataset_YOLOv8"))

from etiquetador_rapido import BBox, HandleType, LienzoEtiquetado, ItemEntrada
import Dividir


def test_bbox_creacion_y_clamping():
    # Caja válida
    b = BBox(class_id=0, xc=0.5, yc=0.5, w=0.2, h=0.2)
    assert b.class_id == 0
    assert abs(b.xc - 0.5) < 1e-5
    assert abs(b.w - 0.2) < 1e-5
    assert not b.is_suelo(0.68)

    # Caja que se sale de los límites debe quedar recortada
    b_out = BBox(class_id=0, xc=1.2, yc=-0.5, w=0.8, h=0.8).clamped()
    assert 0.0 <= b_out.xc <= 1.0
    assert 0.0 <= b_out.yc <= 1.0
    assert b_out.xc + b_out.w / 2.0 <= 1.0 + 1e-5
    assert b_out.yc - b_out.h / 2.0 >= 0.0 - 1e-5


def test_bbox_suelo_deadlift():
    b_suelo = BBox(class_id=0, xc=0.5, yc=0.75, w=0.15, h=0.15)
    assert b_suelo.is_suelo(0.68) is True

    b_alto = BBox(class_id=0, xc=0.5, yc=0.30, w=0.15, h=0.15)
    assert b_alto.is_suelo(0.68) is False


def test_bbox_yolo_serialization():
    b = BBox(class_id=0, xc=0.450000, yc=0.720000, w=0.180000, h=0.200000)
    linea = b.to_yolo_line()
    assert linea.startswith("0 0.450000 0.720000 0.180000 0.200000")

    b_rec = BBox.from_yolo_line(linea)
    assert b_rec is not None
    assert b_rec.class_id == 0
    assert abs(b_rec.xc - 0.45) < 1e-4
    assert abs(b_rec.yc - 0.72) < 1e-4
    assert abs(b_rec.w - 0.18) < 1e-4
    assert abs(b_rec.h - 0.20) < 1e-4


def test_bbox_conversion_px():
    img_w, img_h = 1920, 1080
    rect = QRectF(100.0, 200.0, 300.0, 400.0)
    b = BBox.from_rect_px(rect, img_w, img_h)

    assert 0.0 <= b.xc <= 1.0
    assert 0.0 <= b.yc <= 1.0

    rect_rec = b.to_rect_px(img_w, img_h)
    assert abs(rect_rec.x() - rect.x()) < 1.0
    assert abs(rect_rec.y() - rect.y()) < 1.0
    assert abs(rect_rec.width() - rect.width()) < 1.0
    assert abs(rect_rec.height() - rect.height()) < 1.0


def test_bbox_aspect_ratio():
    img_w, img_h = 1000, 1000
    b = BBox(class_id=0, xc=0.5, yc=0.5, w=0.2, h=0.2)
    assert abs(b.aspect_ratio_px(img_w, img_h) - 1.0) < 1e-5

    # Disco achatado o vertical
    b_vert = BBox(class_id=0, xc=0.5, yc=0.5, w=0.1, h=0.2)
    assert abs(b_vert.aspect_ratio_px(img_w, img_h) - 0.5) < 1e-5


def test_fondo_negativo_yolo(tmp_path: Path):
    # Formato YOLO para fondo negativo: archivo de 0 bytes
    lbl_file = tmp_path / "fondo_001.txt"
    lbl_file.write_text("", encoding="utf-8")

    assert lbl_file.stat().st_size == 0
    # Al leerlo no debe generar ninguna caja
    lineas = [l for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    cajas = [BBox.from_yolo_line(l) for l in lineas]
    assert len(cajas) == 0


def test_tiradores_caja():
    lienzo = LienzoEtiquetado()
    rect = QRectF(100, 100, 200, 200)
    handles = lienzo.get_handles_caja(rect)

    assert len(handles) == 8
    assert HandleType.TOP_LEFT in handles
    assert HandleType.BOTTOM_RIGHT in handles
    assert HandleType.TOP_MID in handles

    # Tirador superior izquierdo debe estar centrado alrededor de (100, 100)
    h_tl = handles[HandleType.TOP_LEFT]
    assert abs(h_tl.center().x() - 100.0) < 1.0
    assert abs(h_tl.center().y() - 100.0) < 1.0


def test_dividir_collect_pool_con_staging(tmp_path: Path, monkeypatch):
    # Crear estructura mínima simulada para testear collect_pool
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    sin_etq_dir = tmp_path / "images_sin_etiquetar"

    for s in ("train", "val", "test"):
        (img_dir / s).mkdir(parents=True)
        (lbl_dir / s).mkdir(parents=True)
    sin_etq_dir.mkdir(parents=True)

    # 1 imagen en train
    (img_dir / "train" / "vid_test1_lift_f00010.jpg").write_text("dummy")
    (lbl_dir / "train" / "vid_test1_lift_f00010.txt").write_text("0 0.5 0.7 0.1 0.1")

    # 1 imagen nueva en images_sin_etiquetar con su label en labels/train
    (sin_etq_dir / "vid_test2_lift_f00020.jpg").write_text("dummy")
    (lbl_dir / "train" / "vid_test2_lift_f00020.txt").write_text("0 0.5 0.3 0.1 0.1")

    monkeypatch.setattr(Dividir, "IMAGES", img_dir)
    monkeypatch.setattr(Dividir, "LABELS", lbl_dir)
    monkeypatch.setattr(Dividir, "BASE", tmp_path)

    pool = Dividir.collect_pool()
    assert "vid_test1_lift" in pool
    assert "vid_test2_lift" in pool
    assert len(pool["vid_test1_lift"]) == 1
    assert len(pool["vid_test2_lift"]) == 1


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    print("Ejecutando pruebas de etiquetado manual...")
    test_bbox_creacion_y_clamping()
    print("  ✓ test_bbox_creacion_y_clamping")
    test_bbox_suelo_deadlift()
    print("  ✓ test_bbox_suelo_deadlift")
    test_bbox_yolo_serialization()
    print("  ✓ test_bbox_yolo_serialization")
    test_bbox_conversion_px()
    print("  ✓ test_bbox_conversion_px")
    test_bbox_aspect_ratio()
    print("  ✓ test_bbox_aspect_ratio")

    with tempfile.TemporaryDirectory() as td:
        test_fondo_negativo_yolo(Path(td))
    print("  ✓ test_fondo_negativo_yolo")

    test_tiradores_caja()
    print("  ✓ test_tiradores_caja")

    class Monkey:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    with tempfile.TemporaryDirectory() as td:
        test_dividir_collect_pool_con_staging(Path(td), Monkey())
    print("  ✓ test_dividir_collect_pool_con_staging")

    print("\n¡Todas las pruebas pasaron exitosamente!")

