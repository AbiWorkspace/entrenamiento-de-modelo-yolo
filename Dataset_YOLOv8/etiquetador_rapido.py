#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etiquetador_rapido.py — Infraestructura de etiquetado manual de frames de video para YOLOv8/YOLO26 (VBT).

Características principales:
  - Navegación y etiquetado manual rápido con atajos de teclado de una sola tecla (Espacio/D, A, F, X, P, Supr, Z, C).
  - Canvas interactivo con Zoom suave centrado en cursor (rueda del ratón) y Paneo (clic derecho o botón central).
  - Creación de cajas delimitadoras (clic y arrastre), selección, desplazamiento (drag & drop) y redimensionado
    con 8 tiradores (handles) de precisión en bordes y esquinas.
  - Soporte de muestras negativas (hard negatives): botón/atajo [F] o [N] para "Fondo puro / Sin disco" (archivo .txt vacío).
  - Descarte de fotogramas borrosos/inútiles con [X] (se mueven a `_descartadas/` para no ensuciar el dataset).
  - Detección geométrica en tiempo real:
      * Resaltado visual en cian para Zona Suelo (yc >= 0.68, Deadlift / Peso Muerto) para cumplir Gate 2 (>=25%).
      * Monitoreo en vivo de relación de aspecto (w/h) con advertencia si se desvía del rango olímpico [0.65, 1.45].
  - Sugerencia asistida opcional con IA [P]: ejecuta inferencia con `best.pt` para pre-rellenar la caja y permitir
    ajuste manual en segundos.
  - Modos de trabajo integrados:
      1. 'sin_etiquetar': fotogramas extraídos pendientes en `images_sin_etiquetar/`.
      2. 'cola': imágenes de `cola_revision_humana.txt` (frames de baja confianza o zona suelo).
      3. 'sospechosas': cajas marcadas en `etiquetas_sospechosas.json`.
      4. 'dataset': revisión/edición de particiones existentes (`images/train`, `val`, `test`).
      5. 'video': carga directa de video MP4 con scrubber de fotogramas y captura de keyframes al vuelo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("etiquetador_rapido")

DIRECTORIO_ACTUAL: Path = Path(__file__).resolve().parent
RAIZ_PROYECTO: Path = DIRECTORIO_ACTUAL.parent
CARPETA_IMAGENES_SIN_ETIQUETAR: Path = DIRECTORIO_ACTUAL / "images_sin_etiquetar"
CARPETA_IMAGENES_BASE: Path = DIRECTORIO_ACTUAL / "images"
CARPETA_LABELS_BASE: Path = DIRECTORIO_ACTUAL / "labels"
CARPETA_DESCARTADAS: Path = DIRECTORIO_ACTUAL / "_descartadas"
COLA_REVISION: Path = DIRECTORIO_ACTUAL / "cola_revision_humana.txt"
REPORTE_SOSPECHOSAS: Path = DIRECTORIO_ACTUAL / "etiquetas_sospechosas.json"
MODELO_DEFAULT: Path = DIRECTORIO_ACTUAL / "best.pt"
CARPETA_VIDEOS: Path = RAIZ_PROYECTO / "Videos para entrenamiento"


# ==============================================================================
# Modelos de datos
# ==============================================================================

@dataclass
class BBox:
    """Caja delimitadora normalizada en formato YOLO: class_id xc yc w h [0, 1]."""
    class_id: int = 0
    xc: float = 0.5
    yc: float = 0.5
    w: float = 0.1
    h: float = 0.1

    def to_rect_px(self, img_w: float, img_h: float) -> QRectF:
        """Convierte coordenadas normalizadas de centro a QRectF en píxeles (esquina sup. izq)."""
        x1 = (self.xc - self.w / 2.0) * img_w
        y1 = (self.yc - self.h / 2.0) * img_h
        return QRectF(x1, y1, self.w * img_w, self.h * img_h)

    @classmethod
    def from_rect_px(cls, rect: QRectF, img_w: float, img_h: float, class_id: int = 0) -> BBox:
        """Crea BBox normalizado a partir de QRectF en píxeles."""
        if img_w <= 0 or img_h <= 0:
            return cls(class_id=class_id, xc=0.5, yc=0.5, w=0.1, h=0.1)
        xc = (rect.x() + rect.width() / 2.0) / img_w
        yc = (rect.y() + rect.height() / 2.0) / img_h
        w = rect.width() / img_w
        h = rect.height() / img_h
        return cls(class_id=class_id, xc=xc, yc=yc, w=w, h=h).clamped()

    def clamped(self) -> BBox:
        """Garantiza que la caja se mantenga estrictamente dentro de [0.0, 1.0]."""
        w = max(0.005, min(1.0, float(self.w)))
        h = max(0.005, min(1.0, float(self.h)))
        half_w = w / 2.0
        half_h = h / 2.0
        xc = max(half_w, min(1.0 - half_w, float(self.xc)))
        yc = max(half_h, min(1.0 - half_h, float(self.yc)))
        return BBox(class_id=self.class_id, xc=xc, yc=yc, w=w, h=h)

    def to_yolo_line(self) -> str:
        b = self.clamped()
        return f"{b.class_id} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}\n"

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional[BBox]:
        partes = line.strip().split()
        if len(partes) >= 5:
            try:
                cid = int(partes[0])
                xc, yc, w, h = (float(v) for v in partes[1:5])
                return cls(class_id=cid, xc=xc, yc=yc, w=w, h=h).clamped()
            except ValueError:
                return None
        return None

    def aspect_ratio_px(self, img_w: float, img_h: float) -> float:
        w_px = self.w * img_w
        h_px = self.h * img_h
        return (w_px / h_px) if h_px > 0 else 1.0

    def is_suelo(self, umbral: float = 0.68) -> bool:
        return self.yc >= umbral


class HandleType:
    NONE = 0
    TOP_LEFT = 1
    TOP_MID = 2
    TOP_RIGHT = 3
    MID_RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM_MID = 6
    BOTTOM_LEFT = 7
    MID_LEFT = 8
    BODY = 9


@dataclass
class ItemEntrada:
    """Representa una imagen del dataset disponible para revisión/etiquetado."""
    ruta_imagen: Path
    ruta_label: Path
    video_id: str = ""
    tiene_label: bool = False
    es_fondo: bool = False
    es_suelo: bool = False
    cajas: list[BBox] = field(default_factory=list)


# ==============================================================================
# Canvas Interactivo (Zoom, Paneo, Edición de Cajas y Tiradores)
# ==============================================================================

class LienzoEtiquetado(QWidget):
    """Widget de dibujo con aceleración gráfica, zoom suave y tiradores de edición."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: #11111b;")

        self.pixmap_original: Optional[QPixmap] = None
        self.img_w: int = 0
        self.img_h: int = 0

        # Transformación de vista
        self.zoom: float = 1.0
        self.pan: QPointF = QPointF(0, 0)

        # Cajas y edición
        self.cajas: list[BBox] = []
        self.caja_seleccionada: int = -1
        self.caja_hover: int = -1
        self.handle_hover: int = HandleType.NONE
        self.handle_activo: int = HandleType.NONE

        # Estados de ratón
        self.modo_arrastre: str = "ninguno"  # "dibujar", "mover", "redimensionar", "pan"
        self.punto_inicio_mouse: QPointF = QPointF(0, 0)
        self.punto_actual_mouse: QPointF = QPointF(0, 0)
        self.pos_mouse_canvas: QPointF = QPointF(0, 0)

        # Historial deshacer
        self.pila_deshacer: list[list[BBox]] = []

        # Parámetros visuales
        self.tam_handle: float = 8.0

    def cargar_pixmap(self, pixmap: QPixmap, cajas: Optional[list[BBox]] = None, reset_vista: bool = True):
        self.pixmap_original = pixmap
        self.img_w = pixmap.width() if pixmap else 0
        self.img_h = pixmap.height() if pixmap else 0
        self.cajas = [b.clamped() for b in cajas] if cajas else []
        self.caja_seleccionada = -1
        self.caja_hover = -1
        self.handle_hover = HandleType.NONE
        self.handle_activo = HandleType.NONE
        self.modo_arrastre = "ninguno"
        self.pila_deshacer.clear()

        if reset_vista:
            self.reset_zoom()
        else:
            self.update()

    def reset_zoom(self):
        """Ajusta la imagen para encajar perfectamente en la ventana."""
        self.pan = QPointF(0, 0)
        self.zoom = 1.0
        self.update()

    def get_geom_imagen(self) -> Optional[QRectF]:
        """Calcula el rectángulo en píxeles del widget donde se proyecta la imagen."""
        if not self.pixmap_original or self.img_w <= 0 or self.img_h <= 0:
            return None

        w_disp = float(self.width())
        h_disp = float(self.height())

        scale_base = min(w_disp / self.img_w, h_disp / self.img_h)
        scale_final = scale_base * self.zoom

        ancho = self.img_w * scale_final
        alto = self.img_h * scale_final

        x = (w_disp - ancho) / 2.0 + self.pan.x()
        y = (h_disp - alto) / 2.0 + self.pan.y()

        return QRectF(x, y, ancho, alto)

    def pantalla_a_norm(self, p: QPointF) -> Optional[tuple[float, float]]:
        geom = self.get_geom_imagen()
        if not geom or geom.width() <= 0 or geom.height() <= 0:
            return None
        u = (p.x() - geom.left()) / geom.width()
        v = (p.y() - geom.top()) / geom.height()
        return max(0.0, min(1.0, u)), max(0.0, min(1.0, v))

    def norm_a_pantalla(self, u: float, v: float) -> Optional[QPointF]:
        geom = self.get_geom_imagen()
        if not geom:
            return None
        x = geom.left() + u * geom.width()
        y = geom.top() + v * geom.height()
        return QPointF(x, y)

    def caja_a_rect_pantalla(self, bbox: BBox) -> Optional[QRectF]:
        geom = self.get_geom_imagen()
        if not geom:
            return None
        w_px = bbox.w * geom.width()
        h_px = bbox.h * geom.height()
        x1_px = geom.left() + (bbox.xc - bbox.w / 2.0) * geom.width()
        y1_px = geom.top() + (bbox.yc - bbox.h / 2.0) * geom.height()
        return QRectF(x1_px, y1_px, w_px, h_px)

    def get_handles_caja(self, rect: QRectF) -> dict[int, QRectF]:
        s = self.tam_handle
        hs = s / 2.0
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        cx = x + w / 2.0
        cy = y + h / 2.0

        return {
            HandleType.TOP_LEFT: QRectF(x - hs, y - hs, s, s),
            HandleType.TOP_MID: QRectF(cx - hs, y - hs, s, s),
            HandleType.TOP_RIGHT: QRectF(x + w - hs, y - hs, s, s),
            HandleType.MID_RIGHT: QRectF(x + w - hs, cy - hs, s, s),
            HandleType.BOTTOM_RIGHT: QRectF(x + w - hs, y + h - hs, s, s),
            HandleType.BOTTOM_MID: QRectF(cx - hs, y + h - hs, s, s),
            HandleType.BOTTOM_LEFT: QRectF(x - hs, y + h - hs, s, s),
            HandleType.MID_LEFT: QRectF(x - hs, cy - hs, s, s),
        }

    def _guardar_deshacer(self):
        self.pila_deshacer.append([BBox(b.class_id, b.xc, b.yc, b.w, b.h) for b in self.cajas])
        if len(self.pila_deshacer) > 30:
            self.pila_deshacer.pop(0)

    def deshacer(self):
        if self.pila_deshacer:
            self.cajas = self.pila_deshacer.pop()
            self.caja_seleccionada = -1
            self.update()

    def limpiar(self):
        if self.cajas:
            self._guardar_deshacer()
            self.cajas = []
            self.caja_seleccionada = -1
            self.update()

    def eliminar_seleccionada(self):
        if 0 <= self.caja_seleccionada < len(self.cajas):
            self._guardar_deshacer()
            self.cajas.pop(self.caja_seleccionada)
            self.caja_seleccionada = -1
            self.update()

    # --------------------------------------------------------------------------
    # Eventos de Ratón y Rueda
    # --------------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        nuevo_zoom = max(0.2, min(20.0, self.zoom * factor))

        geom = self.get_geom_imagen()
        if geom:
            pos_m = event.position()
            u = (pos_m.x() - geom.left()) / geom.width()
            v = (pos_m.y() - geom.top()) / geom.height()

            self.zoom = nuevo_zoom
            nuevo_geom = self.get_geom_imagen()
            if nuevo_geom:
                dx = pos_m.x() - (nuevo_geom.left() + u * nuevo_geom.width())
                dy = pos_m.y() - (nuevo_geom.top() + v * nuevo_geom.height())
                self.pan += QPointF(dx, dy)
        else:
            self.zoom = nuevo_zoom

        self.update()

    def mousePressEvent(self, event):
        pos = event.position()
        self.punto_inicio_mouse = pos
        self.punto_actual_mouse = pos

        # Paneo con clic central o clic derecho
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.modo_arrastre = "pan"
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # 1. Verificar si clic en tirador de la caja seleccionada
            if 0 <= self.caja_seleccionada < len(self.cajas):
                rect_sel = self.caja_a_rect_pantalla(self.cajas[self.caja_seleccionada])
                if rect_sel:
                    handles = self.get_handles_caja(rect_sel)
                    for h_type, h_rect in handles.items():
                        if h_rect.contains(pos):
                            self._guardar_deshacer()
                            self.modo_arrastre = "redimensionar"
                            self.handle_activo = h_type
                            return

            # 2. Verificar si clic dentro de alguna caja (seleccionar y/o mover)
            for idx in reversed(range(len(self.cajas))):
                rect_caja = self.caja_a_rect_pantalla(self.cajas[idx])
                if rect_caja and rect_caja.contains(pos):
                    self._guardar_deshacer()
                    self.caja_seleccionada = idx
                    self.modo_arrastre = "mover"
                    self.handle_activo = HandleType.BODY
                    self.update()
                    return

            # 3. Clic fuera: deseleccionar y comenzar dibujo de nueva caja
            geom = self.get_geom_imagen()
            if geom and geom.contains(pos):
                self.caja_seleccionada = -1
                self.modo_arrastre = "dibujar"
                self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.pos_mouse_canvas = pos

        if self.modo_arrastre == "pan":
            delta = pos - self.punto_actual_mouse
            self.pan += delta
            self.punto_actual_mouse = pos
            self.update()
            return

        if self.modo_arrastre == "dibujar":
            self.punto_actual_mouse = pos
            self.update()
            return

        if self.modo_arrastre == "mover" and 0 <= self.caja_seleccionada < len(self.cajas):
            norm_actual = self.pantalla_a_norm(pos)
            norm_prev = self.pantalla_a_norm(self.punto_actual_mouse)
            if norm_actual and norm_prev:
                du = norm_actual[0] - norm_prev[0]
                dv = norm_actual[1] - norm_prev[1]
                b = self.cajas[self.caja_seleccionada]
                b.xc += du
                b.yc += dv
                self.cajas[self.caja_seleccionada] = b.clamped()
            self.punto_actual_mouse = pos
            self.update()
            return

        if self.modo_arrastre == "redimensionar" and 0 <= self.caja_seleccionada < len(self.cajas):
            norm_pt = self.pantalla_a_norm(pos)
            if norm_pt:
                u, v = norm_pt
                b = self.cajas[self.caja_seleccionada]
                x1 = b.xc - b.w / 2.0
                y1 = b.yc - b.h / 2.0
                x2 = b.xc + b.w / 2.0
                y2 = b.yc + b.h / 2.0

                h = self.handle_activo
                if h == HandleType.TOP_LEFT:
                    x1, y1 = min(u, x2 - 0.005), min(v, y2 - 0.005)
                elif h == HandleType.TOP_MID:
                    y1 = min(v, y2 - 0.005)
                elif h == HandleType.TOP_RIGHT:
                    x2, y1 = max(u, x1 + 0.005), min(v, y2 - 0.005)
                elif h == HandleType.MID_RIGHT:
                    x2 = max(u, x1 + 0.005)
                elif h == HandleType.BOTTOM_RIGHT:
                    x2, y2 = max(u, x1 + 0.005), max(v, y1 + 0.005)
                elif h == HandleType.BOTTOM_MID:
                    y2 = max(v, y1 + 0.005)
                elif h == HandleType.BOTTOM_LEFT:
                    x1, y2 = min(u, x2 - 0.005), max(v, y1 + 0.005)
                elif h == HandleType.MID_LEFT:
                    x1 = min(u, x2 - 0.005)

                nw = max(0.005, x2 - x1)
                nh = max(0.005, y2 - y1)
                nxc = x1 + nw / 2.0
                nyc = y1 + nh / 2.0
                self.cajas[self.caja_seleccionada] = BBox(b.class_id, nxc, nyc, nw, nh).clamped()

            self.punto_actual_mouse = pos
            self.update()
            return

        # Cursor dinámico cuando no se arrastra
        self._actualizar_cursor_hover(pos)
        self.update()

    def _actualizar_cursor_hover(self, pos: QPointF):
        geom = self.get_geom_imagen()
        if not geom or not geom.contains(pos):
            self.setCursor(Qt.ArrowCursor)
            return

        # Tirador seleccionado
        if 0 <= self.caja_seleccionada < len(self.cajas):
            rect_sel = self.caja_a_rect_pantalla(self.cajas[self.caja_seleccionada])
            if rect_sel:
                handles = self.get_handles_caja(rect_sel)
                for h_type, h_rect in handles.items():
                    if h_rect.contains(pos):
                        self._asignar_cursor_handle(h_type)
                        return

        # Cajas existentes
        for idx in reversed(range(len(self.cajas))):
            r = self.caja_a_rect_pantalla(self.cajas[idx])
            if r and r.contains(pos):
                self.setCursor(Qt.SizeAllCursor)
                return

        self.setCursor(Qt.CrossCursor)

    def _asignar_cursor_handle(self, h_type: int):
        if h_type in (HandleType.TOP_LEFT, HandleType.BOTTOM_RIGHT):
            self.setCursor(Qt.SizeFDiagCursor)
        elif h_type in (HandleType.TOP_RIGHT, HandleType.BOTTOM_LEFT):
            self.setCursor(Qt.SizeBDiagCursor)
        elif h_type in (HandleType.TOP_MID, HandleType.BOTTOM_MID):
            self.setCursor(Qt.SizeVerCursor)
        elif h_type in (HandleType.MID_LEFT, HandleType.MID_RIGHT):
            self.setCursor(Qt.SizeHorCursor)

    def mouseReleaseEvent(self, event):
        if self.modo_arrastre == "dibujar":
            geom = self.get_geom_imagen()
            if geom:
                p1 = self.punto_inicio_mouse
                p2 = event.position()
                x_min, x_max = min(p1.x(), p2.x()), max(p1.x(), p2.x())
                y_min, y_max = min(p1.y(), p2.y()), max(p1.y(), p2.y())

                w_px = x_max - x_min
                h_px = y_max - y_min

                # Filtrar toques accidentales de menos de 6 píxeles
                if w_px >= 6.0 and h_px >= 6.0:
                    norm_1 = self.pantalla_a_norm(QPointF(x_min, y_min))
                    norm_2 = self.pantalla_a_norm(QPointF(x_max, y_max))
                    if norm_1 and norm_2:
                        u1, v1 = norm_1
                        u2, v2 = norm_2
                        nw = max(0.005, u2 - u1)
                        nh = max(0.005, v2 - v1)
                        nxc = u1 + nw / 2.0
                        nyc = v1 + nh / 2.0
                        self._guardar_deshacer()
                        nueva_caja = BBox(class_id=0, xc=nxc, yc=nyc, w=nw, h=nh).clamped()
                        self.cajas.append(nueva_caja)
                        self.caja_seleccionada = len(self.cajas) - 1

        self.modo_arrastre = "ninguno"
        self.handle_activo = HandleType.NONE
        self._actualizar_cursor_hover(event.position())
        self.update()

    # --------------------------------------------------------------------------
    # Renderizado Gráfico (QPainter)
    # --------------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.pixmap_original or self.pixmap_original.isNull():
            painter.setPen(QColor("#a6adc8"))
            painter.setFont(QFont("sans-serif", 13))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No hay imagen seleccionada.\nUsa el panel superior para seleccionar modo o video.",
            )
            return

        geom = self.get_geom_imagen()
        if not geom:
            return

        # 1. Dibujar Imagen Escalada
        painter.drawPixmap(
            int(geom.x()),
            int(geom.y()),
            int(geom.width()),
            int(geom.height()),
            self.pixmap_original,
        )

        # 2. Dibujar Retícula de Suelo (yc = 0.68) como guía para Deadlift
        y_suelo_px = geom.top() + 0.68 * geom.height()
        pen_suelo = QPen(QColor(137, 220, 235, 70), 1.0, Qt.DashLine)
        painter.setPen(pen_suelo)
        painter.drawLine(int(geom.left()), int(y_suelo_px), int(geom.right()), int(y_suelo_px))
        painter.setFont(QFont("sans-serif", 8, QFont.Bold))
        painter.setPen(QColor(137, 220, 235, 120))
        painter.drawText(int(geom.left() + 6), int(y_suelo_px - 4), "ZONA SUELO (yc ≥ 0.68) — DEADLIFT")

        # 3. Dibujar Bounding Boxes
        for idx, box in enumerate(self.cajas):
            rect = self.caja_a_rect_pantalla(box)
            if not rect:
                continue

            es_sel = (idx == self.caja_seleccionada)
            es_suelo = box.is_suelo(0.68)

            if es_sel:
                pen = QPen(QColor("#f9e2af"), 2.5)  # Amarillo Catppuccin
                brush = QColor(249, 226, 175, 45)
            elif es_suelo:
                pen = QPen(QColor("#89dceb"), 2.2)  # Cian Suelo
                brush = QColor(137, 220, 235, 40)
            else:
                pen = QPen(QColor("#a6e3a1"), 2.0)  # Verde Menta
                brush = QColor(166, 227, 161, 35)

            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawRect(rect)

            # Etiqueta de la caja
            etiqueta = "disco [SUELO]" if es_suelo else "disco"
            painter.setFont(QFont("sans-serif", 9, QFont.Bold))
            tag_color = QColor("#f9e2af" if es_sel else ("#89dceb" if es_suelo else "#a6e3a1"))
            tag_w = 80 if es_suelo else 50
            painter.fillRect(QRectF(rect.left(), rect.top() - 18, tag_w, 18), tag_color)
            painter.setPen(QColor("#11111b"))
            painter.drawText(QRectF(rect.left() + 4, rect.top() - 17, tag_w, 16), Qt.AlignLeft, etiqueta)

            # Tiradores de redimensionamiento si está seleccionada
            if es_sel:
                handles = self.get_handles_caja(rect)
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#11111b"), 1.2))
                for h_rect in handles.values():
                    painter.drawRect(h_rect)

        # 4. Caja en progreso de dibujo
        if self.modo_arrastre == "dibujar":
            p1 = self.punto_inicio_mouse
            p2 = self.punto_actual_mouse
            rx = min(p1.x(), p2.x())
            ry = min(p1.y(), p2.y())
            rw = abs(p1.x() - p2.x())
            rh = abs(p1.y() - p2.y())

            pen_dibujo = QPen(QColor("#f38ba8"), 1.8, Qt.DashLine)
            painter.setPen(pen_dibujo)
            painter.setBrush(QColor(243, 139, 168, 30))
            painter.drawRect(QRectF(rx, ry, rw, rh))

        # 5. Cruz de cursor sutil (Crosshair)
        if geom.contains(self.pos_mouse_canvas) and self.modo_arrastre == "ninguno":
            pen_cross = QPen(QColor(205, 214, 244, 40), 1.0, Qt.DotLine)
            painter.setPen(pen_cross)
            painter.drawLine(int(geom.left()), int(self.pos_mouse_canvas.y()), int(geom.right()), int(self.pos_mouse_canvas.y()))
            painter.drawLine(int(self.pos_mouse_canvas.x()), int(geom.top()), int(self.pos_mouse_canvas.x()), int(geom.bottom()))

        # 6. HUD Inferior Izquierdo sobre imagen
        self._dibujar_hud(painter, geom)

    def _dibujar_hud(self, painter: QPainter, geom: QRectF):
        painter.setFont(QFont("monospace", 8))
        norm_m = self.pantalla_a_norm(self.pos_mouse_canvas)
        cursor_txt = f"X:{int(norm_m[0]*self.img_w):04d} Y:{int(norm_m[1]*self.img_h):04d}" if norm_m else "---"
        hud_txt = f"{self.img_w}x{self.img_h}px | Zoom:{int(self.zoom*100)}% | {cursor_txt}"

        if 0 <= self.caja_seleccionada < len(self.cajas):
            b = self.cajas[self.caja_seleccionada]
            ar = b.aspect_ratio_px(self.img_w, self.img_h)
            zona = "SUELO (Deadlift) ✓" if b.is_suelo(0.68) else "NORMAL"
            ar_warn = " ⚠️ aspect!" if not (0.65 <= ar <= 1.45) else ""
            hud_txt += f" | Caja #{self.caja_seleccionada+1}: xc={b.xc:.3f} yc={b.yc:.3f} ar={ar:.2f}{ar_warn} [{zona}]"

        painter.fillRect(QRectF(geom.left() + 6, geom.bottom() - 24, 520, 18), QColor(17, 17, 27, 200))
        painter.setPen(QColor("#cdd6f4"))
        painter.drawText(QRectF(geom.left() + 10, geom.bottom() - 22, 510, 16), Qt.AlignLeft, hud_txt)


# ==============================================================================
# Ventana Principal del Etiquetador
# ==============================================================================

class VentanaEtiquetador(QMainWindow):
    def __init__(
        self,
        modo_inicial: str = "sin_etiquetar",
        ruta_video: Optional[Path] = None,
        ruta_cola: Optional[Path] = None,
        ruta_modelo: Optional[Path] = None,
        auto_sugerir: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("Etiquetador Rápido VBT — Detección y Seguimiento de Discos Olímpicos")
        self.resize(1280, 880)

        self.modo_actual = modo_inicial
        self.ruta_video_actual: Optional[Path] = ruta_video
        self.ruta_cola_actual: Path = ruta_cola or COLA_REVISION
        self.ruta_modelo_yolo: Path = ruta_modelo or MODELO_DEFAULT
        self.auto_sugerir_activo: bool = auto_sugerir

        # Ultralytics lazy-load
        self._modelo_yolo = None

        # Lista de ítems y navegación
        self.items: list[ItemEntrada] = []
        self.items_filtrados: list[int] = []
        self.indice_visual: int = 0

        # Video directo
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_total_frames: int = 0
        self.video_frame_idx: int = 0
        self.video_fps: float = 30.0

        self.init_ui()
        self.configurar_atajos()

        # Cargar modo inicial
        self.cambiar_modo(self.modo_actual)

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #181825; }
            QLabel { color: #cdd6f4; font-family: sans-serif; font-size: 12px; }
            QComboBox, QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton#btn_guardar {
                background-color: #89b4fa;
                color: #11111b;
                border: 1px solid #89b4fa;
                font-size: 13px;
            }
            QPushButton#btn_guardar:hover { background-color: #b4befe; }
            QPushButton#btn_fondo {
                background-color: #fab387;
                color: #11111b;
            }
            QPushButton#btn_fondo:hover { background-color: #f9e2af; }
            QPushButton#btn_descartar {
                background-color: #f38ba8;
                color: #11111b;
            }
            QPushButton#btn_ia {
                background-color: #cba6f7;
                color: #11111b;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
        """)

        widget_central = QWidget()
        layout_principal = QVBoxLayout(widget_central)
        layout_principal.setContentsMargins(12, 10, 12, 10)
        layout_principal.setSpacing(8)

        # ----------------------------------------------------------------------
        # Barra Superior: Selector de Modo, Filtros y Progreso
        # ----------------------------------------------------------------------
        layout_top = QHBoxLayout()

        lbl_modo = QLabel("Modo:")
        lbl_modo.setStyleSheet("font-weight: bold;")
        self.combo_modo = QComboBox()
        self.combo_modo.addItem("📁 Sin Etiquetar (images_sin_etiquetar)", "sin_etiquetar")
        self.combo_modo.addItem("📋 Cola Revisión Humana (cola_revision_humana.txt)", "cola")
        self.combo_modo.addItem("⚠️ Etiquetas Sospechosas (etiquetas_sospechosas.json)", "sospechosas")
        self.combo_modo.addItem("🗂 Dataset Actual (images/train, val, test)", "dataset")
        self.combo_modo.addItem("🎬 Video Directo (.mp4 de Videos para entrenamiento)", "video")
        self.combo_modo.currentIndexChanged.connect(self._on_combo_modo_changed)
        layout_top.addWidget(lbl_modo)
        layout_top.addWidget(self.combo_modo)

        # Filtro por video
        self.combo_filtro_video = QComboBox()
        self.combo_filtro_video.addItem("Todos los videos", "")
        self.combo_filtro_video.currentIndexChanged.connect(self._aplicar_filtro)
        layout_top.addWidget(self.combo_filtro_video)

        # Filtro de estado
        self.combo_filtro_estado = QComboBox()
        self.combo_filtro_estado.addItems(["Todos", "Solo pendientes", "Solo etiquetados"])
        self.combo_filtro_estado.currentIndexChanged.connect(self._aplicar_filtro)
        layout_top.addWidget(self.combo_filtro_estado)

        layout_top.addStretch()

        self.lbl_stats = QLabel("Cargando...")
        self.lbl_stats.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        layout_top.addWidget(self.lbl_stats)

        layout_principal.addLayout(layout_top)

        # ----------------------------------------------------------------------
        # Barra de Navegación de Video Directo (oculta si no está en modo video)
        # ----------------------------------------------------------------------
        self.frame_video_bar = QFrame()
        self.frame_video_bar.setStyleSheet("background-color: #1e1e2e; border-radius: 6px; padding: 4px;")
        layout_video = QHBoxLayout(self.frame_video_bar)
        layout_video.setContentsMargins(8, 4, 8, 4)

        self.btn_elegir_video = QPushButton("📂 Seleccionar Video...")
        self.btn_elegir_video.clicked.connect(self._dialogo_elegir_video)
        layout_video.addWidget(self.btn_elegir_video)

        self.lbl_video_info = QLabel("Ningún video cargado")
        layout_video.addWidget(self.lbl_video_info)

        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.valueChanged.connect(self._on_slider_video_changed)
        layout_video.addWidget(self.slider_video, stretch=1)

        self.btn_vid_step_back = QPushButton("◀ (-1)")
        self.btn_vid_step_back.clicked.connect(lambda: self._navegar_frame_video(-1))
        layout_video.addWidget(self.btn_vid_step_back)

        self.btn_vid_step_fwd = QPushButton("▶ (+1)")
        self.btn_vid_step_fwd.clicked.connect(lambda: self._navegar_frame_video(1))
        layout_video.addWidget(self.btn_vid_step_fwd)

        self.btn_vid_keyframe = QPushButton("⏩ Siguiente Movimiento")
        self.btn_vid_keyframe.clicked.connect(self._buscar_siguiente_keyframe_movimiento)
        layout_video.addWidget(self.btn_vid_keyframe)

        self.frame_video_bar.setVisible(False)
        layout_principal.addWidget(self.frame_video_bar)

        # ----------------------------------------------------------------------
        # Barra de Información del Fotograma Actual
        # ----------------------------------------------------------------------
        layout_info_frame = QHBoxLayout()
        self.lbl_frame_nombre = QLabel("Fotograma actual: ---")
        self.lbl_frame_nombre.setFont(QFont("sans-serif", 11, QFont.Bold))
        layout_info_frame.addWidget(self.lbl_frame_nombre)

        layout_info_frame.addStretch()

        self.progreso_suelo = QLabel("Zona Suelo: 0% [Gate 2 Target: ≥25%]")
        self.progreso_suelo.setStyleSheet("color: #89dceb; font-weight: bold;")
        layout_info_frame.addWidget(self.progreso_suelo)

        layout_principal.addLayout(layout_info_frame)

        # ----------------------------------------------------------------------
        # Lienzo Central
        # ----------------------------------------------------------------------
        self.lienzo = LienzoEtiquetado()
        layout_principal.addWidget(self.lienzo, stretch=1)

        # ----------------------------------------------------------------------
        # Barra Inferior de Herramientas y Acciones Rápidas
        # ----------------------------------------------------------------------
        layout_bottom = QHBoxLayout()
        layout_bottom.setSpacing(6)

        self.btn_anterior = QPushButton("◀ Anterior [A]")
        self.btn_anterior.clicked.connect(self.anterior)
        layout_bottom.addWidget(self.btn_anterior)

        self.btn_deshacer = QPushButton("↩ Deshacer [Z]")
        self.btn_deshacer.clicked.connect(self.lienzo.deshacer)
        layout_bottom.addWidget(self.btn_deshacer)

        self.btn_del_box = QPushButton("🗑 Eliminar Caja [Supr]")
        self.btn_del_box.clicked.connect(self.lienzo.eliminar_seleccionada)
        layout_bottom.addWidget(self.btn_del_box)

        self.btn_limpiar = QPushButton("🧹 Limpiar [C]")
        self.btn_limpiar.clicked.connect(self.lienzo.limpiar)
        layout_bottom.addWidget(self.btn_limpiar)

        self.btn_fondo = QPushButton("⬛ Fondo / Sin Disco [F / N]")
        self.btn_fondo.setObjectName("btn_fondo")
        self.btn_fondo.clicked.connect(self.marcar_como_fondo)
        layout_bottom.addWidget(self.btn_fondo)

        self.btn_descartar = QPushButton("❌ Descartar [X]")
        self.btn_descartar.setObjectName("btn_descartar")
        self.btn_descartar.clicked.connect(self.descartar_fotograma)
        layout_bottom.addWidget(self.btn_descartar)

        self.btn_ia = QPushButton("🤖 Sugerir con YOLO [P]")
        self.btn_ia.setObjectName("btn_ia")
        self.btn_ia.clicked.connect(self.sugerir_con_yolo)
        layout_bottom.addWidget(self.btn_ia)

        self.chk_auto_ia = QCheckBox("Auto-IA")
        self.chk_auto_ia.setChecked(self.auto_sugerir_activo)
        self.chk_auto_ia.stateChanged.connect(self._on_chk_auto_ia_changed)
        layout_bottom.addWidget(self.chk_auto_ia)

        self.btn_zoom_fit = QPushButton("🔍 Encajar [0]")
        self.btn_zoom_fit.clicked.connect(self.lienzo.reset_zoom)
        layout_bottom.addWidget(self.btn_zoom_fit)

        layout_bottom.addStretch()

        self.btn_guardar_siguiente = QPushButton("Guardar y Siguiente ▶ [Espacio / D]")
        self.btn_guardar_siguiente.setObjectName("btn_guardar")
        self.btn_guardar_siguiente.clicked.connect(self.guardar_y_siguiente)
        layout_bottom.addWidget(self.btn_guardar_siguiente)

        layout_principal.addLayout(layout_bottom)
        self.setCentralWidget(widget_central)

        self.statusBar().showMessage(
            "Arrastra el ratón para marcar discos olímpicos. Rueda para Zoom, Clic derecho para Paneo. "
            "Atajos: [Espacio/D] Guardar | [A] Anterior | [F] Fondo | [P] Sugerir IA | [Supr] Borrar caja | [0] Encajar zoom."
        )

    def configurar_atajos(self):
        """Atajos de teclado de alta velocidad para maximizar la productividad."""
        QShortcut(QKeySequence("D"), self, self.guardar_y_siguiente)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.guardar_y_siguiente)
        QShortcut(QKeySequence("A"), self, self.anterior)
        QShortcut(QKeySequence("F"), self, self.marcar_como_fondo)
        QShortcut(QKeySequence("N"), self, self.marcar_como_fondo)
        QShortcut(QKeySequence("X"), self, self.descartar_fotograma)
        QShortcut(QKeySequence("P"), self, self.sugerir_con_yolo)
        QShortcut(QKeySequence("I"), self, self.sugerir_con_yolo)
        QShortcut(QKeySequence("Z"), self, self.lienzo.deshacer)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.lienzo.deshacer)
        QShortcut(QKeySequence("C"), self, self.lienzo.limpiar)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.lienzo.eliminar_seleccionada)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.lienzo.eliminar_seleccionada)
        QShortcut(QKeySequence("0"), self, self.lienzo.reset_zoom)
        QShortcut(QKeySequence("+"), self, lambda: self._zoom_paso(1.2))
        QShortcut(QKeySequence("-"), self, lambda: self._zoom_paso(1.0 / 1.2))

    def _zoom_paso(self, factor: float):
        self.lienzo.zoom = max(0.2, min(20.0, self.lienzo.zoom * factor))
        self.lienzo.update()

    def _on_chk_auto_ia_changed(self, state):
        self.auto_sugerir_activo = (state == Qt.Checked)

    # --------------------------------------------------------------------------
    # Gestión de Modos y Carga de Datos
    # --------------------------------------------------------------------------

    def _on_combo_modo_changed(self, index: int):
        modo = self.combo_modo.itemData(index)
        self.cambiar_modo(modo)

    def cambiar_modo(self, modo: str):
        self.modo_actual = modo
        # Sincronizar combo si se invocó programáticamente
        idx = self.combo_modo.findData(modo)
        if idx >= 0 and self.combo_modo.currentIndex() != idx:
            self.combo_modo.setCurrentIndex(idx)

        self.frame_video_bar.setVisible(modo == "video")

        if modo == "video":
            if self.ruta_video_actual and self.ruta_video_actual.is_file():
                self._cargar_video(self.ruta_video_actual)
            else:
                self._buscar_primer_video_disponible()
            return

        # Modos basados en imágenes
        self._cargar_items_modo(modo)

    def _cargar_items_modo(self, modo: str):
        self.items.clear()
        CARPETA_LABELS_BASE.mkdir(parents=True, exist_ok=True)
        (CARPETA_LABELS_BASE / "train").mkdir(parents=True, exist_ok=True)

        vid_re = re.compile(r"^(?P<vid>vid_[^_]+_[^_]+)")

        if modo == "sin_etiquetar":
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            if CARPETA_IMAGENES_SIN_ETIQUETAR.is_dir():
                rutas = sorted([
                    p for p in CARPETA_IMAGENES_SIN_ETIQUETAR.iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                ])
                for p in rutas:
                    m = vid_re.match(p.stem)
                    vid = m.group("vid") if m else "vid_misc"
                    # Buscar label candidata
                    lbl = None
                    for candidata in (
                        CARPETA_LABELS_BASE / "train" / f"{p.stem}.txt",
                        CARPETA_IMAGENES_SIN_ETIQUETAR / f"{p.stem}.txt",
                    ):
                        if candidata.is_file():
                            lbl = candidata
                            break
                    if lbl is None:
                        lbl = CARPETA_LABELS_BASE / "train" / f"{p.stem}.txt"

                    self.items.append(ItemEntrada(ruta_imagen=p, ruta_label=lbl, video_id=vid))

        elif modo == "cola":
            if self.ruta_cola_actual.is_file():
                lineas = [l.strip() for l in self.ruta_cola_actual.read_text(encoding="utf-8").splitlines() if l.strip()]
                for l in lineas:
                    p = Path(l)
                    if p.is_file():
                        m = vid_re.match(p.stem)
                        vid = m.group("vid") if m else "vid_misc"
                        lbl = CARPETA_LABELS_BASE / "train" / f"{p.stem}.txt"
                        self.items.append(ItemEntrada(ruta_imagen=p, ruta_label=lbl, video_id=vid))

        elif modo == "sospechosas":
            if REPORTE_SOSPECHOSAS.is_file():
                try:
                    data = json.loads(REPORTE_SOSPECHOSAS.read_text(encoding="utf-8"))
                    sospechosas = data.get("sospechosas", [])
                    for s in sospechosas:
                        rel = s.get("archivo", "")
                        lbl_path = DIRECTORIO_ACTUAL / rel
                        img_path = lbl_path.with_suffix(".jpg")
                        if not img_path.is_file():
                            # intentar en images/train
                            img_path = CARPETA_IMAGENES_BASE / "train" / f"{lbl_path.stem}.jpg"
                        if img_path.is_file():
                            m = vid_re.match(img_path.stem)
                            vid = m.group("vid") if m else "vid_misc"
                            self.items.append(ItemEntrada(ruta_imagen=img_path, ruta_label=lbl_path, video_id=vid))
                except Exception as e:
                    logger.warning("Error leyendo %s: %s", REPORTE_SOSPECHOSAS, e)

        elif modo == "dataset":
            for split in ("train", "val", "test"):
                split_dir = CARPETA_IMAGENES_BASE / split
                if split_dir.is_dir():
                    for p in sorted(split_dir.glob("vid_*")):
                        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            m = vid_re.match(p.stem)
                            vid = m.group("vid") if m else "vid_misc"
                            lbl = CARPETA_LABELS_BASE / split / f"{p.stem}.txt"
                            self.items.append(ItemEntrada(ruta_imagen=p, ruta_label=lbl, video_id=vid))

        # Actualizar lista de videos en el combo filtro
        videos_unicos = sorted({item.video_id for item in self.items if item.video_id})
        self.combo_filtro_video.blockSignals(True)
        self.combo_filtro_video.clear()
        self.combo_filtro_video.addItem("Todos los videos", "")
        for v in videos_unicos:
            self.combo_filtro_video.addItem(v, v)
        self.combo_filtro_video.blockSignals(False)

        self._aplicar_filtro()

    def _aplicar_filtro(self):
        vid_filtro = self.combo_filtro_video.currentData() or ""
        estado_filtro = self.combo_filtro_estado.currentText()

        self.items_filtrados = []
        for idx, it in enumerate(self.items):
            if vid_filtro and it.video_id != vid_filtro:
                continue

            tiene_lbl = it.ruta_label.is_file()
            if estado_filtro == "Solo pendientes" and tiene_lbl:
                continue
            if estado_filtro == "Solo etiquetados" and not tiene_lbl:
                continue

            self.items_filtrados.append(idx)

        self.indice_visual = 0
        self._actualizar_estadisticas_globales()
        self.mostrar_item_actual()

    def _actualizar_estadisticas_globales(self):
        total = len(self.items)
        etiquetadas = sum(1 for it in self.items if it.ruta_label.is_file())
        pendientes = total - etiquetadas

        # Calcular porcentaje de zona suelo sobre las etiquetadas
        total_cajas = 0
        cajas_suelo = 0
        for it in self.items:
            if it.ruta_label.is_file():
                try:
                    lineas = it.ruta_label.read_text(encoding="utf-8").splitlines()
                    for ln in lineas:
                        b = BBox.from_yolo_line(ln)
                        if b:
                            total_cajas += 1
                            if b.is_suelo(0.68):
                                cajas_suelo += 1
                except Exception:
                    pass

        pct_suelo = (cajas_suelo / total_cajas * 100.0) if total_cajas > 0 else 0.0
        color_suelo = "#a6e3a1" if pct_suelo >= 25.0 else "#fab387"

        self.lbl_stats.setText(f"✓ {etiquetadas} etiquetadas | ⏳ {pendientes} pendientes | Total: {total}")
        self.progreso_suelo.setText(f"Zona Suelo (Deadlift): {cajas_suelo}/{total_cajas} ({pct_suelo:.1f}%) [Gate 2: ≥25%]")
        self.progreso_suelo.setStyleSheet(f"color: {color_suelo}; font-weight: bold;")

    # --------------------------------------------------------------------------
    # Visualización y Navegación de Imágenes
    # --------------------------------------------------------------------------

    def mostrar_item_actual(self):
        if self.modo_actual == "video":
            return

        if not self.items_filtrados:
            self.lbl_frame_nombre.setText("No hay imágenes que coincidan con el filtro actual.")
            self.lienzo.cargar_pixmap(QPixmap(), [])
            return

        idx_real = self.items_filtrados[self.indice_visual]
        item = self.items[idx_real]

        # Cargar pixmap
        pixmap = QPixmap(str(item.ruta_imagen))

        # Cargar cajas existentes si hay label
        cajas: list[BBox] = []
        if item.ruta_label.is_file():
            try:
                lineas = item.ruta_label.read_text(encoding="utf-8").splitlines()
                for ln in lineas:
                    b = BBox.from_yolo_line(ln)
                    if b:
                        cajas.append(b)
            except Exception as e:
                logger.warning("Error leyendo %s: %s", item.ruta_label, e)

        # Si no hay cajas y auto-IA está activado, sugerir
        if not cajas and self.auto_sugerir_activo and not item.ruta_label.is_file():
            cajas = self._inferir_cajas_con_yolo(item.ruta_imagen)

        self.lienzo.cargar_pixmap(pixmap, cajas, reset_vista=False)

        total_filtro = len(self.items_filtrados)
        estado_lbl = "✓ Guardado" if item.ruta_label.is_file() else "⏳ Pendiente"
        self.lbl_frame_nombre.setText(
            f"Foto [{self.indice_visual + 1} de {total_filtro}] ({estado_lbl}): {item.ruta_imagen.name}"
        )

    def guardar_y_siguiente(self):
        if self.modo_actual == "video":
            self._guardar_frame_video_actual()
            self._navegar_frame_video(1)
            return

        if not self.items_filtrados:
            return

        idx_real = self.items_filtrados[self.indice_visual]
        item = self.items[idx_real]

        # Guardar cajas
        lineas = [b.to_yolo_line() for b in self.lienzo.cajas]
        contenido = "".join(lineas)

        # Destino oficial: labels/train e images/train
        dest_lbl = CARPETA_LABELS_BASE / "train" / f"{item.ruta_imagen.stem}.txt"
        dest_img = CARPETA_IMAGENES_BASE / "train" / item.ruta_imagen.name

        dest_lbl.write_text(contenido, encoding="utf-8")
        if item.ruta_imagen != dest_img and not dest_img.exists():
            shutil.copy2(str(item.ruta_imagen), str(dest_img))

        # También guardar copia en ruta local del ítem si era diferente
        if item.ruta_label != dest_lbl:
            item.ruta_label.write_text(contenido, encoding="utf-8")

        self.statusBar().showMessage(
            f"✓ Guardado: {item.ruta_imagen.name} ({len(self.lienzo.cajas)} cajas)", 3000
        )
        self._actualizar_estadisticas_globales()

        # Avanzar
        if self.indice_visual < len(self.items_filtrados) - 1:
            self.indice_visual += 1
            self.mostrar_item_actual()
        else:
            QMessageBox.information(
                self, "Fin de lista", "¡Has completado el etiquetado de todos los fotogramas del filtro actual!"
            )

    def anterior(self):
        if self.modo_actual == "video":
            self._navegar_frame_video(-1)
            return

        if self.indice_visual > 0:
            self.indice_visual -= 1
            self.mostrar_item_actual()

    def marcar_como_fondo(self):
        """Muestra negativa oficial YOLO: archivo .txt de 0 bytes."""
        self.lienzo.cajas = []
        self.guardar_y_siguiente()
        self.statusBar().showMessage("⬛ Marcado como Fondo Puro / Sin Disco (hard negative)", 3000)

    def descartar_fotograma(self):
        """Mueve un fotograma defectuoso a _descartadas para evitar sesgos."""
        if self.modo_actual == "video" or not self.items_filtrados:
            return

        idx_real = self.items_filtrados[self.indice_visual]
        item = self.items[idx_real]

        CARPETA_DESCARTADAS.mkdir(parents=True, exist_ok=True)
        dest_img = CARPETA_DESCARTADAS / item.ruta_imagen.name
        if item.ruta_imagen.exists():
            shutil.move(str(item.ruta_imagen), str(dest_img))

        if item.ruta_label.exists():
            item.ruta_label.unlink()

        self.statusBar().showMessage(f"❌ Fotograma descartado y movido a _descartadas: {item.ruta_imagen.name}", 3000)
        self.items.pop(idx_real)
        self._aplicar_filtro()

    # --------------------------------------------------------------------------
    # Modo Video Directo (.mp4)
    # --------------------------------------------------------------------------

    def _buscar_primer_video_disponible(self):
        formatos = {".mp4", ".mov", ".avi"}
        if CARPETA_VIDEOS.is_dir():
            videos = sorted([p for p in CARPETA_VIDEOS.iterdir() if p.suffix.lower() in formatos])
            if videos:
                self._cargar_video(videos[0])
                return
        self.lbl_video_info.setText("No se encontraron videos en 'Videos para entrenamiento/'")

    def _dialogo_elegir_video(self):
        archivo, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Video para Etiquetado",
            str(CARPETA_VIDEOS if CARPETA_VIDEOS.is_dir() else RAIZ_PROYECTO),
            "Archivos de Video (*.mp4 *.mov *.avi *.mkv)",
        )
        if archivo:
            self._cargar_video(Path(archivo))

    def _cargar_video(self, ruta: Path):
        if self.video_cap:
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(str(ruta))
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", f"No se pudo abrir el video:\n{ruta}")
            return

        self.ruta_video_actual = ruta
        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_frame_idx = 0

        self.slider_video.blockSignals(True)
        self.slider_video.setRange(0, max(0, self.video_total_frames - 1))
        self.slider_video.setValue(0)
        self.slider_video.blockSignals(False)

        self._mostrar_frame_video_actual()

    def _mostrar_frame_video_actual(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_idx)
        ret, frame_bgr = self.video_cap.read()
        if not ret or frame_bgr is None:
            return

        h, w, ch = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Comprobar si ya existe guardado
        vid_id = self._obtener_video_id(self.ruta_video_actual)
        nombre_stem = f"{vid_id}_f{self.video_frame_idx:05d}"
        lbl_existente = CARPETA_LABELS_BASE / "train" / f"{nombre_stem}.txt"

        cajas: list[BBox] = []
        if lbl_existente.is_file():
            try:
                for ln in lbl_existente.read_text(encoding="utf-8").splitlines():
                    b = BBox.from_yolo_line(ln)
                    if b:
                        cajas.append(b)
            except Exception:
                pass

        if not cajas and self.auto_sugerir_activo:
            cajas = self._inferir_cajas_con_yolo(frame_bgr)

        self.lienzo.cargar_pixmap(pixmap, cajas, reset_vista=False)

        segundos = self.video_frame_idx / self.video_fps
        segundos_totales = self.video_total_frames / self.video_fps
        self.lbl_video_info.setText(
            f"{self.ruta_video_actual.name} | Frame {self.video_frame_idx}/{self.video_total_frames} "
            f"({segundos:.1f}s / {segundos_totales:.1f}s)"
        )
        self.lbl_frame_nombre.setText(f"Video Directo: {nombre_stem}.jpg")

    def _on_slider_video_changed(self, valor: int):
        self.video_frame_idx = valor
        self._mostrar_frame_video_actual()

    def _navegar_frame_video(self, delta: int):
        if not self.video_cap:
            return
        nuevo = max(0, min(self.video_total_frames - 1, self.video_frame_idx + delta))
        if nuevo != self.video_frame_idx:
            self.video_frame_idx = nuevo
            self.slider_video.setValue(nuevo)

    def _buscar_siguiente_keyframe_movimiento(self):
        """Avanza buscando un cambio de movimiento o aceleración cinemática."""
        if not self.video_cap:
            return

        self.statusBar().showMessage("Buscando próximo keyframe cinemático...")
        QApplication.processEvents()

        curr_idx = self.video_frame_idx
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
        ret, f_ant = self.video_cap.read()
        if not ret:
            return
        g_ant = cv2.cvtColor(cv2.resize(f_ant, (320, 320)), cv2.COLOR_BGR2GRAY)

        idx = curr_idx + 4
        while idx < self.video_total_frames - 5:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f_act = self.video_cap.read()
            if not ret:
                break
            g_act = cv2.cvtColor(cv2.resize(f_act, (320, 320)), cv2.COLOR_BGR2GRAY)
            diff = float(np.mean(cv2.absdiff(g_act, g_ant)))
            if diff >= 2.5:
                self.video_frame_idx = idx
                self.slider_video.setValue(idx)
                self.statusBar().showMessage(f"Keyframe detectado en frame {idx} (diff={diff:.1f})", 3000)
                return
            g_ant = g_act
            idx += 4

        self.statusBar().showMessage("Fin de video alcanzado", 2000)

    def _guardar_frame_video_actual(self):
        if not self.video_cap or not self.ruta_video_actual:
            return

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_idx)
        ret, frame_bgr = self.video_cap.read()
        if not ret:
            return

        vid_id = self._obtener_video_id(self.ruta_video_actual)
        nombre_stem = f"{vid_id}_f{self.video_frame_idx:05d}"
        dest_img = CARPETA_IMAGENES_BASE / "train" / f"{nombre_stem}.jpg"
        dest_lbl = CARPETA_LABELS_BASE / "train" / f"{nombre_stem}.txt"

        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dest_img), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        contenido = "".join(b.to_yolo_line() for b in self.lienzo.cajas)
        dest_lbl.write_text(contenido, encoding="utf-8")

        self.statusBar().showMessage(f"✓ Capturado y guardado al dataset: {dest_img.name}", 3000)

    @staticmethod
    def _obtener_video_id(ruta: Path) -> str:
        digest = hashlib.sha1(str(ruta).encode("utf-8")).hexdigest()[:8]
        base = ruta.stem
        base = re.sub(r"\s*\(.*?\)\s*$", "", base)
        base = re.sub(r"~\d+\s*$", "", base)
        limpio = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_").lower()
        return f"vid_{digest}_{limpio or 'video'}"

    # --------------------------------------------------------------------------
    # Pre-Anotación Asistida con YOLO (IA)
    # --------------------------------------------------------------------------

    def sugerir_con_yolo(self):
        """Ejecuta inferencia con el modelo YOLOv8 para predecir cajas en el fotograma."""
        cajas: list[BBox] = []
        if self.modo_actual == "video":
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_idx)
                ret, frame_bgr = self.video_cap.read()
                if ret:
                    cajas = self._inferir_cajas_con_yolo(frame_bgr)
        else:
            if self.items_filtrados:
                idx_real = self.items_filtrados[self.indice_visual]
                cajas = self._inferir_cajas_con_yolo(self.items[idx_real].ruta_imagen)

        if cajas:
            self.lienzo._guardar_deshacer()
            self.lienzo.cajas = cajas
            self.lienzo.caja_seleccionada = 0
            self.lienzo.update()
            self.statusBar().showMessage(f"🤖 YOLO sugirió {len(cajas)} caja(s). Ajusta los tiradores si es necesario.", 4000)
        else:
            self.statusBar().showMessage("🤖 YOLO no detectó discos con confianza suficiente. Márcalo manualmente.", 4000)

    def _inferir_cajas_con_yolo(self, fuente: Path | np.ndarray) -> list[BBox]:
        if not self.ruta_modelo_yolo.is_file():
            # Buscar alternativas
            for alt in (
                DIRECTORIO_ACTUAL / "yolo26n.pt",
                DIRECTORIO_ACTUAL / "yolov8n.pt",
                RAIZ_PROYECTO / "weights" / "best.pt",
            ):
                if alt.is_file():
                    self.ruta_modelo_yolo = alt
                    break

        if not self.ruta_modelo_yolo.is_file():
            return []

        if self._modelo_yolo is None:
            try:
                from ultralytics import YOLO
                self._modelo_yolo = YOLO(str(self.ruta_modelo_yolo))
            except Exception as e:
                logger.warning("No se pudo cargar modelo YOLO: %s", e)
                return []

        try:
            inp = str(fuente) if isinstance(fuente, Path) else fuente
            res = self._modelo_yolo.predict(inp, conf=0.15, imgsz=1024, verbose=False)
            if not res or len(res[0].boxes) == 0:
                return []

            boxes = res[0].boxes
            xywhn = boxes.xywhn.cpu().numpy()
            cajas_out: list[BBox] = []
            for b in xywhn:
                xc, yc, w, h = (float(v) for v in b)
                cajas_out.append(BBox(class_id=0, xc=xc, yc=yc, w=w, h=h).clamped())
            return cajas_out
        except Exception as e:
            logger.warning("Error en inferencia de pre-anotación: %s", e)
            return []


# ==============================================================================
# Entrada CLI
# ==============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Etiquetador Manual y Asistido de Discos Olímpicos (YOLO VBT)."
    )
    parser.add_argument(
        "--mode",
        choices=["sin_etiquetar", "cola", "sospechosas", "dataset", "video"],
        default="sin_etiquetar",
        help="Modo inicial de trabajo (default: sin_etiquetar).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Ruta a un archivo MP4 para abrir directamente en modo video.",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="Ruta a archivo de cola (por defecto: cola_revision_humana.txt).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Modelo YOLO para pre-anotaciones sugeridas (default: best.pt).",
    )
    parser.add_argument(
        "--auto-suggest",
        action="store_true",
        help="Activar sugerencias automáticas de IA al navegar.",
    )
    args = parser.parse_args(argv)

    modo = "video" if args.video else args.mode

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Etiquetador Rapido VBT")

    ventana = VentanaEtiquetador(
        modo_inicial=modo,
        ruta_video=args.video,
        ruta_cola=args.queue,
        ruta_modelo=args.model,
        auto_sugerir=args.auto_suggest,
    )
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
