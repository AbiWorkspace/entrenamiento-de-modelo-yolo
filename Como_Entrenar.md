# Como Entrenar — Guía de ejecución completa (YOLO VBT)

Pipeline completo: entrenamiento → exportación → verificación E2E.
Hardware: **AMD Radeon RX 7900 XTX** (ROCm). Python: `~/.venv-rocm/bin/python`.

> Toda la ejecución se hace desde esta carpeta de proyecto:
> `~/Escritorio/Entrenamiento de modelo yolo` (el nombre termina en espacio).

```bash
cd "/home/abimael/Escritorio/Entrenamiento de modelo yolo "
PY=~/.venv-rocm/bin/python
```

---

## Paso 0 — Liberar VRAM: detener Ollama (manual)

Ollama consume ~7 GiB de VRAM y compite con el entrenamiento.

```bash
# Detener el servicio (elige la opción que aplique a tu instalación):
systemctl --user stop ollama
# o bien:
pkill -f ollama

# Verificar que no queda proceso:
pgrep -a ollama || echo "Ollama detenido ✓"
```

Confirmar VRAM libre antes de empezar:

```bash
rocm-smi | grep -i vram
```

Debe mostrar ~20+ GiB libres. Si Ollama se reinicia solo, mantenerlo detenido durante todo el entrenamiento.

---

## Fase Previa — Extracción y Etiquetado Manual de Fotogramas

Para agregar nuevos videos o enriquecer el dataset con fotogramas etiquetados a mano (especialmente Zona Suelo / Peso Muerto):

### 1. Extracción de Keyframes (Fase 1)

```bash
# Opción A: Extraer keyframes conscientes de blur de todos los videos
$PY "Videos para entrenamiento/extraer_frames.py" --max-frames 45 --workers 8

# Opción B: Extraer keyframes de un único video específico
$PY "Videos para entrenamiento/extraer_frames.py" --video "Videos para entrenamiento/20261019_105121.mp4" --max-frames 45
```

### 2. Etiquetado Manual Asistido (GUI PySide6)

Lanza el entorno de etiquetado interactivo:

```bash
# Modo 1: Fotogramas pendientes en images_sin_etiquetar (por defecto)
$PY etiquetar.py
# o bien:
./etiquetar.sh

# Modo 2: Revisar cola de incertidumbre (imágenes dudosas o suelo)
$PY etiquetar.py --cola

# Modo 3: Revisar alertas de auditoría
$PY etiquetar.py --sospechosas

# Modo 4: Navegar y etiquetar directamente desde el video MP4 (sin extraer previamente)
$PY etiquetar.py --video "Videos para entrenamiento/20261019_105121.mp4"

# Activar sugerencias de IA asistida al cambiar de fotograma:
$PY etiquetar.py --auto-ia
```

#### Atajos de Teclado del Etiquetador:
| Tecla | Acción |
|---|---|
| `Espacio` o `D` | **Guardar y Siguiente** (asigna label YOLO `0 xc yc w h`) |
| `A` | **Anterior** |
| `F` o `N` | **Fondo puro / Sin disco** (genera hard negative oficial de 0 bytes) |
| `X` | **Descartar fotograma** (mueve a `_descartadas/` si está inservible) |
| `P` o `I` | **Sugerir con YOLO (IA)** (pre-anota con `best.pt` para afinar bordes) |
| `Supr` o `Del` | **Eliminar caja** seleccionada |
| `Z` o `Ctrl+Z` | **Deshacer** última modificación |
| `C` | **Limpiar** todas las cajas de la imagen |
| `Rueda Ratón` | **Zoom suave** centrado en la posición del cursor |
| `Clic Der. / Central` | **Paneo** (arrastrar para mover la imagen en zoom) |
| `0` | **Encajar imagen** / Resetear zoom |

> [!TIP]
> **Alineación de bordes y 8 tiradores:** Haz clic sobre cualquier caja para seleccionarla y arrastra sus 8 tiradores en bordes/esquinas para ajustar el contorno exacto del disco. El recuadro cambia a cian cuando entra en la **Zona Suelo ($y_c \ge 0.68$)**, indicando que suma al cupo de Peso Muerto para superar el **Gate 2 ($\ge 25\%$)**.

### 3. Auditoría y Partición Anti-Fuga (Fases 2 y 3)

Una vez completado el etiquetado manual:

```bash
# Auditar consistencia geométrica y distribución vertical
$PY Dataset_YOLOv8/auditar_dataset.py --report

# Particionar automáticamente a nivel de video_id (Cero Data Leakage)
$PY Dataset_YOLOv8/Dividir.py --train 0.70 --val 0.20 --test 0.10 --verify-no-leakage
```

---

## Paso 1 — Entrenamiento (Fase 5)

```bash
$PY Dataset_YOLOv8/Entrenar.py
```

Duración estimada: depende de épocas/batch (ver `Dataset_YOLOv8/hyp_vbt.yaml`).
Salidas esperadas en `Dataset_YOLOv8/`:
- `best.pt`, `last.pt` (modelos)
- métricas de validación por época (logs)

### Gate 5 — umbrales de producción

Al terminar, verificar que las métricas finales superen:

| Métrica   | Mínimo |
|-----------|--------|
| F1        | ≥ 0.94 |
| mAP50     | ≥ 0.96 |
| mAP50-95  | ≥ 0.82 |

Si no se alcanzan: revisar `hyp_vbt.yaml` (lr, patience) o calidad de dataset antes de continuar.
Las métricas pasan a `Dataset_YOLOv8/metricas_produccion.json` al superar el gate.

---

## Paso 2 — Exportación ONNX + benchmark + deploy (Fase 7)

```bash
$PY Dataset_YOLOv8/Exportar_Produccion.py --deploy
```

Qué hace:
1. Exporta `best.pt` → `best.onnx`.
2. Ejecuta benchmark de inferencia (ONNX vs PyTorch).
3. Deploy atómico del artefacto a la carpeta de servicios.

Verificar que `Dataset_YOLOv8/best.onnx` exista y no esté vacío:

```bash
ls -lh Dataset_YOLOv8/best.onnx
```

> NCNN (mobile): pendiente — requiere `pnnx` en PATH. Es un follow-up, no bloquea.

---

## Paso 3 — Test E2E del motor VBT (Fase 8)

Procesa hasta 6 videos reales de `Videos para entrenamiento/` con el modelo
entrenado y valida:

- estructura del dict de retorno (`estado`, `reps`, `v_promedio_global`, …)
- KPIs de robustez de tracking: `cobertura_frames ∈ [0,1]`,
  `reinicios_por_hueco ≥ 0` (int)
- `max(v_peak) ≤ 2.2 m/s` (tope físico razonable)
- video renderizado generado y no vacío

Modo standalone (genera `reporte_e2e_vbt.json`):

```bash
$PY tests/test_e2e_vbt.py --max-videos 6 --report reporte_e2e_vbt.json
```

Modo pytest (requiere `pip install pytest` si no está en el venv):

```bash
$PY -m pytest tests/test_e2e_vbt.py -v
```

Notas:
- Los videos se etiquetan cíclicamente [Sentadilla, Press de Banca, Peso Muerto]×2
  (no hay ground-truth de reps; la validación es estructural + rangos).
- Salida renderizada temporal en `/tmp/opencode/e2e_vbt/`.
- El envío de WhatsApp está stubbeado (se loguea, no envía).
- Cada video tarda varios minutos (tracking GPU + render HUD + ffmpeg);
  6 videos ≈ decenas de minutos. Dejar correr sin interrumpir.

---

## Orden resumido (una línea por paso)

1. `pkill -f ollama` → confirmar VRAM libre con `rocm-smi`.
2. `$PY Dataset_YOLOv8/Entrenar.py` → chequear Gate 5 (F1≥0.94, mAP50≥0.96, mAP50-95≥0.82).
3. `$PY Dataset_YOLOv8/Exportar_Produccion.py --deploy` → verificar `best.onnx`.
4. `$PY tests/test_e2e_vbt.py --max-videos 6 --report reporte_e2e_vbt.json` → revisar reporte JSON.

## Diagnóstico rápido de fallos

| Síntoma | Causa probable | Acción |
|---------|----------------|--------|
| `out of memory` / OOM ROCm | Ollama activo u otro proceso | `pkill -f ollama`, relanzar |
| `data.meta_service` no resuelve al importar | import fuera de contexto del test | usar `tests/test_e2e_vbt.py` (ya stubbeado) |
| `estado: "error"` en E2E, 0 reps | video sin disco visible / ejercicio mal mapeado | ver `mensaje` en reporte; probar otro video |
| ONNX export falla | falta opset/runtime | revisar log de `Exportar_Produccion.py` |
