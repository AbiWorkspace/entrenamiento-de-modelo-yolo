# Reporte técnico — Sistema VBT YOLO (detección de disco olímpico)

Fecha: 19/10/2026

## 1. Stack actual

| Componente | Detalle |
|---|---|
| Modelo base | YOLOv8n (`yolov8n.pt`), fine-tuning; también existe `yolo26n.pt` |
| Clases | 1 clase: `disco` (data.yaml: nc=1, names=['disco']) |
| Dataset | 1.061 imágenes etiquetadas (848 train / 213 val) en `Dataset_YOLOv8/` |
| Entrenamiento | `Entrenar.py`: imgsz=1024, batch=16, epochs=50, workers=8, GPU (device=0), validación de GPU al arranque |
| Tracking | ByteTrack (`bytetrack.yaml`), `max_det=3`, `conf=0.15`, `iou_nms=0.45` |
| Inferencia | FP16 en GPU; despliegue móvil vía NCNN (`best_ncnn_model/`) |
| Servicio | `vbt_service.py`: descarga video de Meta → motor VBT → sube resultado |
| Motor | `Creardataset.py`: detección + tracking, cinemática de barra, máquina de estados (reposo / empuje-concéntrica), RIR/RPE/1RM estimado |
| Videos | ~45 MP4 en `Videos para entrenamiento/` (ago–sep 2026) |

## 2. Pipeline del dataset

1. **Extracción** — `extraer_frames.py` (antes: 1 de cada N frames; ahora: detección de keyframes por movimiento)
2. **Pre-etiquetado IA** — `auto_etiquetar.py`: usa el `best.pt` anterior con umbral 0.45 y genera previews
3. **Etiquetado manual** — `etiquetador_rapido.py` (GUI PySide6, cajas por arrastre, coordenadas normalizadas)
4. **División** — `Dividir.py`: reparte en train/val, mueve sin etiqueta a `images_sin_etiquetar/`
5. **Entrenamiento** — `Entrenar.py` / `Entrenar_Mobile.py`

## 3. Errores encontrados y corregidos en esta sesión

| Problema | Corrección |
|---|---|
| `data.yaml` apuntaba a ruta antigua (`Probable entrenamiento/Dataset_YOLOv8`) que no coincide con la ubicación real | Corregida a `/home/abimael/Escritorio/Entrenamiento de modelo yolo /Dataset_YOLOv8` |
| `classes.txt` con 5 entradas obsoletas (Weight, weight plate, machine, black circular weight…) inconsistentes con data.yaml (1 clase) | Reescrito con la única clase `disco` |
| Extracción de frames ciegamente secuencial (1 cada N) → sesgo por duración del video: sesiones largas dominaban el dataset y los frames casi idénticos se repetían | `extraer_frames.py` reescrito: detección de keyframes por diferencia de frames (umbral 18/255, separación mínima 6 frames, tope 40 frames/video) |

## 4. Sesgos conocidos del dataset

- **Peso muerto / deadlift subrepresentado** en entrenamientos previos → el modelo pierde precisión en esa posición del disco.
- La extracción antigua no era consciente del ejercicio: un video de 3 min aportaba ~36× más frames que uno de 15 s, independientemente de la cantidad real de información (poses distintas).
- Frames casi duplicados (misma pose) inflaban el dataset sin aportar variedad.

## 5. Áreas de mejora pendientes (roadmap)

1. **Etiquetar los nuevos keyframes** con `auto_etiquetar.py` + revisión en `etiquetador_rapido.py`, verificar cobertura de deadlift/peso muerto y posiciones bajas/altas del disco.
2. **Reentrenar** con `Entrenar.py` tras la ampliación; comparar métricas mAP50-95 por posición (baja/media/alta) antes vs después.
3. **Balanceo por ejercicio**: si algún ejercicio sigue corto, duplicar sus frames o usar `hyp` de aumentos (mosaic/mixup ya disponibles en ultralytics).
4. **Consolidar nomenclatura**: asegurar que cualquier script que lea `classes.txt` no dependa de las 5 entradas viejas (buscar referencias antes de re-etiquetar).
5. **Validación móvil**: tras el nuevo entrenamiento, repasar `Entrenar_Mobile.py` y exportar NCNN para comparar degradación FP16 vs NCNN en el dispositivo objetivo.
6. **Opcional — tracking**: si ByteTrack pierde el disco en transiciones rápidas, probar `max_det=1` (solo hay una clase y 1 disco relevante por vez) o ajustar `track_buffer`.

## 6. Archivos clave

- `vbt_service.py` — orquestador del servicio
- `Creardataset.py` — motor cinemático + máquina de estados
- `Videos para entrenamiento/extraer_frames.py` — extracción inteligente (reescrita)
- `Dataset_YOLOv8/data.yaml`, `classes.txt` — config del dataset (corregidas)
- `Dataset_YOLOv8/{Entrenar.py, Entrenar_Mobile.py, Dividir.py, auto_etiquetar.py, etiquetador_rapido.py}`
- `Dataset_YOLOv8/best.pt`, `runs/detect/train-1024/weights/best.pt` — checkpoints
