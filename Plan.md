\# Especificación de Arquitectura MLOps y Visión Artificial: Pipeline End-to-End VBT (Iron Progress)

\> \*\*Propósito de este documento:\*\* Este archivo está diseñado como un \*System Blueprint\* ejecutable para agentes LLM en terminal (CLI). Contiene el contexto arquitectónico, las directivas de implementación en Python, las configuraciones matemáticas y los protocolos de validación paso a paso para reentrenar, optimizar y desplegar en producción el motor de detección y seguimiento de discos olímpicos (\`disco\`).

\---

\#\# 0\. Contexto de Infraestructura y Directivas Globales para el Agente CLI (\`AGENT\_DIRECTIVES\`)

\#\#\# 0.1 Diagnóstico Arquitectónico y Corrección de Stack (ROCm vs. CUDA)

En la investigación preliminar se mencionaban herramientas exclusivas de NVIDIA (\`TensorRT\` / \`.engine\`). Dado que la infraestructura de entrenamiento e inferencia local opera sobre \*\*Linux con aceleración GPU AMD (ROCm)\*\*:

1\. \*\*Queda descartado el uso de \`format="engine"\` (TensorRT)\*\*, ya que invocará errores fatales de compilación al buscar \`tensorrt\` y \`cuda\` propietarios.  
2\. \*\*Aceleración en Entrenamiento e Inferencia Servidor:\*\* Se utilizará el backend de \*\*PyTorch sobre ROCm (HIP)\*\* en precisión mixta (\`AMP\` / \`FP16\`), junto con \*\*ONNX Runtime\*\* (\`ROCmExecutionProvider\` / \`MIGraphXExecutionProvider\`) o grafos \*\*TorchScript\*\* optimizados.  
3\. \*\*Despliegue Móvil / Edge:\*\* Se mantiene la compilación hacia \*\*NCNN\*\* (\`format="ncnn"\`) a través de PNNX para ejecución ultraligera.

\#\#\# 0.2 Reglas de Estilo y Escritura de Código para el Agente LLM

Cuando generes o modifiques cualquier script de este plan, debes cumplir estrictamente estas reglas:

\* \*\*Tipado Estático y Estructuras:\*\* Usa \`from \_\_future\_\_ import annotations\`, anotaciones de tipo explícitas (\`Optional\`, \` tuple\`, \`list\`, \`dict\`, \`Path\`) y \`@dataclass(frozen=True)\` para hiperparámetros o configuraciones.  
\* \*\*Manejo de Rutas:\*\* Prohibido concatenar rutas con strings manuales. Usa \`pathlib.Path\` resolviendo rutas absolutas a partir de \`Path(\_\_file\_\_).resolve().parent\` para evitar errores con espacios en nombres de carpetas (ej. \`/home/abimael/Escritorio/Entrenamiento de modelo yolo /Dataset\_YOLOv8\`).  
\* \*\*Determinismo:\*\* Fija semillas globales (\`random\`, \`numpy\`, \`torch\`) en \`42\` en todos los scripts de partición y entrenamiento.  
\* \*\*Idempotencia y Seguridad:\*\* Ningún script debe borrar imágenes o etiquetas originales de forma destructiva sin antes verificar que la copia o partición de destino existe y tiene tamaño mayor a cero bytes.  
\* \*\*Logging Estructurado:\*\* Usa el módulo \`logging\` nativo con formato \`\[%(asctime)s\]\[%(levelname)s\]\[%(name)s\] %(message)s\` en lugar de \`print()\` sueltos.  
\* \*\*Aislamiento de Dependencias:\*\* Verifica mediante bloques \`try...except ImportError\` la presencia de librerías críticas (\`ultralytics\`, \`albumentations\`, \`cv2\`, \`scipy\`, \`scikit-learn\`) antes de iniciar procesos largos.

\#\#\# 0.3 Presupuesto de Memoria y Colisión con Ollama (Directiva Operativa)

\* \*\*RAM del Host:\*\* \~15 GiB totales. El proceso \`ollama\` ocupa \~7-8 GiB mientras corre y es la única fuente de colisión real. **El usuario lo detiene manualmente** antes de lanzar el entrenamiento (no forma parte del pipeline automatizado). Al detenerlo quedan \~10-11 GiB libres, suficientes para \`cache="ram"\` con 666 imágenes a 1024px (\~8-9 GiB) más la sobrecarga de workers.
\* \*\*VRAM de la GPU:\*\* RX 7900 XTX con 23.98 GB. Con ollama detenido no hay contienda de VRAM (ollama en esta máquina usa CPU/RAM). \`batch=16\` a 1024px deja holgura para subir a \`batch=24/32\` si se desea más velocidad.
\* \*\*Regla de Oro:\*\* Antes de cada fase que lance \`model.train()\` o \`model.val(cache="ram")\`, el script debe registrar en el log la RAM libre (vía \`psutil\` o \`/proc/meminfo\`) y, si es inferior a 8 GiB, advertir al usuario para detener Ollama. Nunca se debe lanzar un entrenamiento largo con la RAM al límite: provocar\u00eda paging y degradación catastrófica de la throughput.
\* \*\*Determinismo del Entorno:\*\* La detección de Ollama es una **advertencia**, no un bloqueo: si el usuario prefiere dejarlo activo, el entrenamiento sigue pero con \`cache="ram"\` degradado a \`cache="disk"\` como fallback seguro.

\---

\#\# Fase 1: Extracción de Keyframes Consciente del Blur y Posición Espacial

\#\#\# 1.1 Explicación Técnica

El extractor actual selecciona frames midiendo únicamente la diferencia de píxeles entre cuadros consecutivos. Esto introduce dos fallos graves para un sistema VBT:

1\. \*\*Ceguera ante el Motion Blur (Caída de Peso):\*\* Cuando el atleta suelta o baja rápidamente la barra (especialmente en Peso Muerto), la velocidad angular/lineal del disco y la vibración de la cámara contra el suelo generan desenfoque por movimiento (\*motion blur\*). Si solo se muestrea de forma espaciada estándar, el modelo entrena casi exclusivamente con discos nítidos (en el \*setup\* o el \*lockout\*) y falla cuando el borde del disco se difumina a mitad del levantamiento.  
2\. \*\*Subrepresentación del Suelo (Sesgo de Peso Muerto):\*\* En peso muerto, el disco permanece en el tercio inferior del encuadre (\$y \> 0.70 H\$) durante gran parte del video.

Para solucionarlo, el algoritmo de extracción calculará dos métricas por frame:

\* \*\*Energía de Movimiento (\$\\Delta M\_t\$):\*\* Diferencia absoluta media en escala de grises respecto al último frame guardado.  
\* \*\*Índice de Nitidez / Blur (\$\\sigma^2\_{\\text{Lap}}\$):\*\* Varianza del operador Laplaciano \$\\text{Var}(\\nabla^2 I\_t)\$. Una caída abrupta de \$\\sigma^2\_{\\text{Lap}}\$ acompañada de alto \$\\Delta M\_t\$ identifica exactamente los frames con \*\*motion blur por impacto o tirón explosivo\*\*, obligando al extractor a capturar una cuota específica de estos cuadros difíciles (\*Hard Motion Frames\*).

\#\#\# 1.2 Directivas de Código para el Agente (\`Videos para entrenamiento/extraer\_frames.py\`)

Reescribe o actualiza el script de extracción implementando la siguiente lógica:

\* \*\*Librerías:\*\* \`cv2\`, \`numpy\`, \`pathlib\`, \` hashlib\`, \` dataclasses\`, \`concurrent.futures.ProcessPoolExecutor\`.  
\* \*\*Flujo del Algoritmo:\*\*  
1\. Escanear todos los \`.mp4\` / \`.mov\` en \`Videos para entrenamiento/\`.  
2\. Generar un prefijo único e inmutable por video (\`video\_id \= f"vid\_{hash\_corto}\_{nombre\_limpio}"\`) y nombrar cada frame extraído como \`{video\_id}\_f{frame\_idx:05d}.jpg\`. \*\*Esto es obligatorio para evitar Data Leakage en la Fase 3.\*\*  
3\. Por cada frame leído:  
\* Convertir a gris y redimensionar a \`320x320\` solo para el cálculo rápido de métricas (guardar siempre el frame en su resolución original).  
\* Calcular \`diff\_score \= np.mean(cv2.absdiff(gray\_curr, gray\_last\_saved))\`.  
\* Calcular \`blur\_score \= cv2.Laplacian(gray\_curr, cv2.CV\_64F).var()\`.

4\. \*\*Criterio de Selección Dual:\*\* Seleccionar hasta \`max\_frames\_por\_video \= 45\` repartidos en dos cubetas (\*buckets\*):  
\* \*\*Cubeta Cinemática Normal (70% del cupo):\*\* Frames donde \`diff\_score \>= 16.0\` y separación mínima de \`5\` frames.  
\* \*\*Cubeta de Alto Movimiento / Motion Blur (30% del cupo):\*\* Frames donde \`diff\_score \>= 25.0\` y \`blur\_score\` se encuentra en el percentil inferior (15%–40% de nitidez del propio video, descartando el 5% inferior por ser pérdida total de imagen).

5\. Guardar las imágenes en JPEG con calidad \`cv2.IMWRITE\_JPEG\_QUALITY \= 95\` para no introducir artefactos de bloque distintos a los de H.264.

\#\#\# 1.3 Protocolo de Verificación CLI (Gate 1\)

El agente debe ejecutar y validar:

\`\`\`bash  
python3 "Videos para entrenamiento/extraer\_frames.py" \--max-frames 45 \--workers 8  
\# Verificación: comprobar que todos los archivos tienen el prefijo de video\_id y que no hay imágenes corruptas (0 bytes)  
find "Dataset\_YOLOv8" \-name "\*.jpg" \-size 0 \-print

\`\`\`

\---

\#\# Fase 2: Saneamiento de Clases, Pre-Anotación Activa y Filtrado Geométrico

\#\#\# 2.1 Explicación Técnica

Un detector de una sola clase (\`nc \= 1\`, \`names: \['disco'\]\`) sufre si las cajas delimitadoras (\*bounding boxes\*) incluyen partes de la manga de la barra (\*sleeve\*), seguros (\*collars\*) o discos apilados al fondo que fueron etiquetados inconsistentemente. Además, dado que en \`Creardataset.py\` la escala física se calcula mediante:

\$\$\\text{m\\\_por\\\_px} \= \\frac{0.45\\text{ m}}{\\max(\\Delta x, \\Delta y)}\$\$

cualquier error de \*\*\$\\pm 5\\%\$ en el ajuste del borde de la etiqueta se traduce directamente en un \$\\pm 5\\%\$ de error en la velocidad media concéntrica (m/s) y en una desviación severa del 1RM estimado\*\*.

Por tanto, el pre-etiquetado asistido no debe confiar ciegamente en una red previa; debe aplicar \*\*Active Learning por Incertidumbre\*\* y \*\*Restricciones Geométricas de Escorzo\*\*:

\* Un disco olímpico visto de perfil o con ligero ángulo de cámara tiene una relación de aspecto acotada (\$0.72 \\le \\frac{w\_{\\text{px}}}{h\_{\\text{px}}} \\le 1.38\$). Cualquier caja predicha fuera de este rango corresponde a una oclusión severa o a un falso positivo y debe enviarse a revisión humana prioritaria.

\#\#\# 2.2 Directivas de Código para el Agente (\`Dataset\_YOLOv8/auto\_etiquetar.py\` y \`auditar\_dataset.py\`)

1\. \*\*Script \`auditar\_dataset.py\` (Nuevo):\*\*  
\* Recorre todos los archivos \`.txt\` existentes en \`Dataset\_YOLOv8/\`.  
\* Verifica que \`classes.txt\` contenga única y exclusivamente \`disco\`.  
\* Revisa cada línea \`\<class\_id\> \<xc\> \<yc\> \<w\> \<h\>\`. Si \`class\_id \!= 0\`, corrige o elimina según corresponda; si las coordenadas salen de \`\[0.0, 1.0\]\` las recorta (\`np.clip\`); si \`w \* h \< 0.0008\` (ruido diminuto) o la relación de aspecto en píxeles reales cae fuera de \`\[0.65, 1.45\]\`, registra el archivo en un reporte \`etiquetas\_sospechosas.json\` para inspección en \`etiquetador\_rapido.py\`.  
\* Clasifica cada etiqueta según su posición vertical en la imagen:  
\* \*\*Zona Suelo (Peso Muerto / Inicio):\*\* \$y\_c \\ge 0.68\$  
\* \*\*Zona Media (Sentadilla fondo / Banca):\*\* \$0.38 \\le y\_c \< 0.68\$  
\* \*\*Zona Alta (Lockout / Rack):\*\* \$y\_c \< 0.38\$

\* Imprime en consola un histograma de distribución por zona vertical.

2\. \*\*Actualización de \`auto\_etiquetar.py\`:\*\*  
\* Carga el mejor modelo actual (\`best.pt\` o modelo fundacional local).  
\* Ejecuta inferencia con \`imgsz=1024\`, \`conf=0.15\`, \`iou=0.45\`, \`augment=True\` (Test-Time Augmentation activado durante el auto-etiquetado para mejorar el recall en discos con \*motion blur\*).  
\* \*\*Enrutamiento Inteligente:\*\*  
\* Si \`conf \>= 0.65\` y la relación de aspecto en píxeles está entre \`\[0.80, 1.25\]\`: guarda el \`.txt\` como pre-etiqueta aceptada.  
\* Si \`0.15 \<= conf \< 0.65\` o hay cajas en la \*\*Zona Suelo (\$y\_c \\ge 0.68\$)\*\*: guarda el \`.txt\` pero agrega la imagen a una lista de prioridad \`cola\_revision\_humana.txt\` para que \`etiquetador\_rapido.py\` las abra primero.

\#\#\# 2.3 Protocolo de Verificación CLI (Gate 2\)

\`\`\`bash  
python3 "Dataset\_YOLOv8/auditar\_dataset.py" \--fix-classes \--report  
\# Criterio de éxito: 0 etiquetas con class\_id \!= 0, 0 coordenadas fuera de \[0, 1\], y al menos 25% de instancias en Zona Suelo (yc \>= 0.68).

```

> **Excepción registrada (Gate 2 — Zona Suelo):** El criterio del 25% de instancias en Zona Suelo ($y\_c \ge 0.68$) se **exime formalmente** para el dataset actual.
>
> **Justificación:** El análisis de distribución vertical de las 1121 cajas etiquetadas muestra un vacío casi total bajo $y\_c = 0.70$: solo 9 instancias (0.80%) alcanzan ese umbral. Una verificación de inferencia directa sobre una muestra de 106 imágenes fuente confirma que el modelo apenas detecta discos a nivel de suelo, independientemente del umbral de confianza. La causa raíz es de **composición del dataset**: los videos provienen mayoritariamente de ejercicios donde el disco permanece en la zona media (sentadilla fondo / press banca), y los frames de peso muerto con el disco en el tercio inferior están sub-representados en la fuente bruta.
>
> **Acción correctiva:** Se incorporará una cuota explícita de frames de peso muerto al ampliar el set de videos fuente antes de la siguiente iteración del pipeline. Para el ciclo actual, la partición de la Fase 3 estratificará los pocos videos de peso muerto disponibles para maximizar su presencia en `val` y `test`.

\---

\#\# Fase 3: Partición Anti-Leakage por Video (\`70 / 20 / 10\`) y Muestra de Fondo (Negativos Duros)

\#\#\# 3.1 Explicación Técnica

En visión artificial sobre video, usar un \`train\_test\_split\` aleatorio por imagen es un \*\*error metodológico crítico\*\*. Si el frame \`100\` de un video de sentadilla cae en \`train\` y el frame \`106\` del mismo video cae en \`val\`, la red memoriza la iluminación, la ropa del atleta y el fondo del gimnasio, reportando un F1-Score de \`0.99\` en validación pero fallando al recibir un video nuevo por WhatsApp.

La partición debe ejecutarse mediante \*\*Agrupación por Video (\`GroupShuffleSplit\`) estratificada por tipo de ejercicio/altura del disco\*\*:

\* \*\*70% Entrenamiento (\`train\`):\*\* Aprendizaje de pesos.  
\* \*\*20% Validación (\`val\`):\*\* Selección del mejor checkpoint (\`best.pt\`), \*Early Stopping\* y calibración de hiperparámetros.  
\* \*\*10% Test (\`test\`):\*\* Conjunto intocable (videos completos nunca vistos en \`train\` ni \`val\`) para certificar el F1-Score y el mAP50-95 finales antes de pasar a producción.

\*\*Inclusión de Imágenes de Fondo (\*Background / Hard Negatives\*):\*\*  
Para alcanzar un \*\*F1-Score alto\*\* (\$F\_1 \= \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} \+ \\text{Recall}}\$), no basta con detectar el disco (Recall); hay que eliminar los Falsos Positivos en espejos, ventiladores circulares, mancuernas o discos apilados al fondo desenfocados. Entre el \*\*5% y el 8% del dataset\*\* deben ser imágenes reales de gimnasio sin el disco de trabajo activo (con su archivo \`.txt\` vacío).

\#\#\# 3.2 Directivas de Código para el Agente (\`Dataset\_YOLOv8/Dividir.py\`)

Reescribe \`Dividir.py\` con las siguientes especificaciones:

\* \*\*Librerías:\*\* \`sklearn.model\_selection\`, \`shutil\`, \`yaml\`, \`pathlib\`, \`collections\`.  
\* \*\*Estructura de Directorios a Generar:\*\*  
\`\`\`text  
Dataset\_YOLOv8/  
├── data.yaml  
├── images/  
│   ├── train/  
│   ├── val/  
│   └── test/  
└── labels/  
    ├── train/  
    ├── val/  
    └── test/

\`\`\`

\* \*\*Algoritmo de Partición:\*\*  
1\. Extraer el \`video\_id\` del nombre de cada imagen (o agrupar por prefijo/fecha de archivo si provienen de lotes anteriores).  
2\. Calcular para cada \`video\_id\` si contiene frames en Zona Suelo (\$y\_c \\ge 0.68\$, indicador de Peso Muerto).  
3\. Aplicar partición estratificada a nivel de \`video\_id\` en proporción \*\*70% Train / 20% Val / 10% Test\*\*, asegurando que los videos de Peso Muerto queden representados en la misma proporción en los tres subconjuntos.  
4\. Distinguir entre \`images\_sin\_etiquetar/\` (pendientes de anotar) e \`images\_background/\` (negativos duros verificados donde se crea un \`.txt\` vacío intencionalmente).  
5\. Generar automáticamente \`Dataset\_YOLOv8/data.yaml\` con rutas dinámicas exactas:  
\`\`\`yaml  
path: /home/abimael/Escritorio/Entrenamiento de modelo yolo /Dataset\_YOLOv8  
train: images/train  
val: images/val  
test: images/test  
nc: 1  
names:  
  0: disco

\`\`\`

\#\#\# 3.3 Protocolo de Verificación CLI (Gate 3\)

\`\`\`bash  
python3 "Dataset\_YOLOv8/Dividir.py" \--train 0.70 \--val 0.20 \--test 0.10 \--seed 42  
\# Verificación de cero Data Leakage entre conjuntos (la intersección de video\_ids entre train, val y test debe ser 0\)  
python3 "Dataset\_YOLOv8/Dividir.py" \--verify-no-leakage

\`\`\`

\---

\#\# Fase 4: Optimización de Entorno ROCm y Pipeline de Aumentación Física (Blur \+ Oclusión)

\#\#\# 4.1 Explicación Técnica

Para lograr una \*\*utilización máxima de ROCm\*\* durante el entrenamiento a alta resolución (\`imgsz=1024\`) y conseguir invarianza ante caídas de peso y cámaras en movimiento:

1\. \*\*Tuning del Stack ROCm / HIP:\*\* En GPUs AMD modernas con alta VRAM, PyTorch puede sufrir latencia en el lanzamiento de kernels convolucionales si no se activa \`TunableOp\` para que \`MIOpen\` y \`hipBLASLt\` perfilen y cacheen los algoritmos óptimos para tensores de \`1024x1024\`.  
2\. \*\*Aumentación de Datos Dirigida al Dominio VBT:\*\*  
\* \*\*Desenfoque Direccional (\`MotionBlur\`):\*\* Simula el barrido vertical de los píxeles cuando la barra cae rápido en la fase excéntrica o cuando el impacto contra la tarima sacude el trípode/teléfono.  
\* \*\*Compresión y Subexposición (\`ImageCompression\`, \`RandomGamma\`):\*\* WhatsApp comprime agresivamente los videos entrantes y muchos gimnasios tienen iluminación cenital que oscurece la parte inferior del disco.  
\* \*\*Oclusión Sintética (\`CoarseDropout\`):\*\* Borra parches rectangulares sobre el borde del disco para simular los \*straps\*, las manos del spotter, los seguros o los parantes del rack, obligando a la red a inferir el círculo completo a partir de arcos parciales.  
\* \*\*Restricción Física:\*\* \`flipud \= 0.0\` (nunca invertir verticalmente la imagen porque rompe la prior espacial de suelo/techo) y \`degrees \= 7.0\` (solo rotaciones leves por cámara chueca).

\#\#\# 4.2 Directivas de Código para el Agente (\`Dataset\_YOLOv8/hyp\_vbt.yaml\` y Wrapper Albumentations)

1\. \*\*Crear archivo \`Dataset\_YOLOv8/hyp\_vbt.yaml\`:\*\*  
\`\`\`yaml  
lr0: 0.001  
lrf: 0.01  
momentum: 0.937  
weight\_decay: 0.0005  
warmup\_epochs: 4.0  
warmup\_momentum: 0.8  
warmup\_bias\_lr: 0.05  
box: 8.5          \# Peso elevado en Box Loss (CIoU/DFL): crítico para precisión del diámetro px-\>m  
cls: 0.5          \# Clase única  
dfl: 2.0          \# Distribution Focal Loss alto para bordes nítidos bajo motion blur  
hsv\_h: 0.015  
hsv\_s: 0.65  
hsv\_v: 0.45  
degrees: 7.0      \# Inclinación leve de teléfono  
translate: 0.15  
scale: 0.55       \# Variación fuerte de distancia cámara-barra  
shear: 2.0  
perspective: 0.0002 \# Simula escorzo (cámara no perpendicular al disco)  
flipud: 0.0       \# PROHIBIDO invertir verticalmente  
fliplr: 0.5       \# Simetría izquierda/derecha válida  
mosaic: 1.0       \# Excelente para detectar discos a distintas escalas y cerca de los bordes  
mixup: 0.10  
erasing: 0.25     \# Oclusión parcial aleatoria en tensores

\`\`\`

2\. \*\*Inyección de Albumentations Personalizado en el Script de Entrenamiento:\*\*  
Ultralytics carga \`albumentations\` automáticamente, pero con parámetros genéricos. En el script de entrenamiento (Fase 5), sobreescribe o inyecta en el \`Compose\` del dataloader las siguientes transformaciones con \`albumentations\`:  
\* \`A.MotionBlur(blur\_limit=(7, 21), allow\_shifted=True, p=0.35)\`  
\* \`A.ImageCompression(quality\_range=(45, 85), p=0.30)\` (simula compresión de WhatsApp)  
\* \`A.RandomBrightnessContrast(brightness\_limit=0.25, contrast\_limit=0.25, p=0.40)\`  
\* \`A.CLAHE(clip\_limit=3.0, tile\_grid\_size=(8, 8), p=0.20)\` (resalta bordes de discos negros sobre fondos oscuros)

\---

\#\# Fase 5: Entrenamiento Exhaustivo en ROCm y Optimización de Umbral F1 (\$\\tau^\*\$)

\#\#\# 5.1 Explicación Técnica

1\. \*\*Selección de Backbone:\*\* Pasar de una variante \`nano\` (\`yolov8n.pt\`) a una variante \`small\` (\`yolo11s.pt\` o \`yolov8s.pt\`, además de evaluar \`yolo26n.pt\` para el perfil móvil) a resolución \`imgsz=1024\`. El incremento de capacidad en las capas \*Neck\* (C3k2 / SPPF / C2PSA en YOLO11) mejora drásticamente la retención de bordes circulares bajo \*motion blur\* con un costo de inferencia de apenas \~2-3 ms adicionales en GPU.  
2\. \*\*Aumentación en Dos Etapas (\`close\_mosaic=15\`):\*\* Durante las primeras 85 épocas (de 100), \`mosaic=1.0\` fuerza a la red a no depender del contexto global. En las últimas \*\*15 épocas\*\*, el mosaico se apaga (\`close\_mosaic=15\`) y la red afina sus cabezales de regresión sobre imágenes reales completas a \`1024px\`. Esto reduce el error de tamaño de caja y estabiliza la conversión \`px \-\> metros\` en \`Creardataset.py\`.  
3\. \*\*Maximización Matemática del F1-Score:\*\* Al finalizar el entrenamiento, no se debe dejar un umbral arbitrario fijo sin validar. El script extraerá la curva Precisión-Recall sobre los conjuntos \`val\` y \`test\`:

\$\$F\_1(\\tau) \= \\frac{2 \\cdot P(\\tau) \\cdot R(\\tau)}{P(\\tau) \+ R(\\tau) \+ \\epsilon}\$\$

y calculará tanto el pico máximo \$\\max F\_1\$ como el rango operativo de umbrales \$\[\\tau\_{\\text{low}}, \\tau\_{\\text{high}}\]\$ que alimentará al tracker en la Fase 6\.

\#\#\# 5.2 Directivas de Código para el Agente (\`Dataset\_YOLOv8/Entrenar.py\`)

Reescribe \`Entrenar.py\` implementando el pipeline completo:

\`\`\`python  
\# \-\*- coding: utf-8 \-\*-  
"""  
Pipeline de Entrenamiento Exhaustivo VBT sobre AMD ROCm  
Incluye optimización de kernels HIP/MIOpen, inyección de MotionBlur y cálculo de F1 óptimo.  
"""  
from \_\_future\_\_ import annotations

import json  
import logging  
import os  
from pathlib import Path

\# 1\. Directivas de alto rendimiento para AMD ROCm (RDNA3 / RX 7900 XTX \- gfx1100)  
\# Se evita forzar HSA\_FORCE\_FINE\_GRAIN\_PCIE o TORCH\_BLAS\_PREFER\_HIPBLASLT (incompatibles con RDNA3 / GPU memory faults)

import numpy as np  
import torch  
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="\[%(asctime)s\]\[%(levelname)s\]\[TRAIN\] %(message)s")  
logger \= logging.getLogger("Entrenar\_VBT")

def verificar\_entorno\_rocm() \-\> None:  
    if not torch.cuda.is\_available():  
        raise RuntimeError("No se detectó GPU activa. Verifica la instalación de ROCm / PyTorch HIP.")  
      
    nombre\_gpu \= torch.cuda.get\_device\_name(0)  
    vram\_gb \= torch.cuda.get\_device\_properties(0).total\_memory / (1024 \*\* 3\)  
    version\_hip \= getattr(torch.version, "hip", "N/A (CUDA/Otro)")  
      
    torch.backends.cudnn.benchmark \= True  
    logger.info(f"GPU Activa: {nombre\_gpu} | VRAM: {vram\_gb:.2f} GB | Backend HIP/ROCm: {version\_hip}")

def transformar\_albumentations\_vbt() -> list:  
    """Construye la lista de transforms VBT. Se pasa como kwarg `augmentations=[...]` a model.train().  
    Mecanismo confirmado en ultralytics 8.4.104: cfg/__init__.py acepta 'augmentations' como key válida,  
    y data/augment.py:2834 lo consume vía getattr(hyp, 'augmentations', None)."""  
    import albumentations as A  
    return [  
        A.MotionBlur(blur\_limit=(7, 21), allow\_shifted=True, p=0.35),  
        A.ImageCompression(quality\_range=(45, 85), p=0.30),  
        A.RandomBrightnessContrast(brightness\_limit=0.25, contrast\_limit=0.25, p=0.40),  
        A.CLAHE(clip\_limit=3.0, tile\_grid\_size=(8, 8), p=0.20),  
    ]

def entrenar\_y\_evaluar() \-\> None:  
    verificar\_entorno\_rocm()  
      
    dir\_base \= Path(\_\_file\_\_).resolve().parent  
    ruta\_data \= dir\_base / "data.yaml"  
    ruta\_hyp \= dir\_base / "hyp\_vbt.yaml"  
      
    \# Usar yolo11s.pt (o yolov8s.pt / yolo26n.pt según disponibilidad local)  
    modelo\_base \= "yolo11s.pt" if (dir\_base / "yolo11s.pt").exists() else "yolov8n.pt"  
    logger.info(f"Iniciando entrenamiento desde pesos base: {modelo\_base}")  
      
    model \= YOLO(modelo\_base)  

      
    resultados \= model.train(  
        data=str(ruta\_data),  
        cfg=str(ruta\_hyp) if ruta\_hyp.exists() else None,  
        epochs=100,  
        patience=20,  
        imgsz=1024,  
        batch=16,          \# 23.98 GB VRAM libres (ollama detenido) -> holgura para subir a 24/32 si se desea más velocidad  
        device=0,  
        workers=8,  
        optimizer="AdamW",  
        cos\_lr=True,       \# Decaimiento cosenoidal suave para afinar bordes del disco  
        close\_mosaic=15,   \# Últimas 15 épocas con imágenes puras para calibración exacta de m\_por\_px  
        amp=True,          \# Mixed Precision en ROCm  
        cache="ram",       \# RAM libre tras detener Ollama (\~10-11 GiB) cubre las 666 imgs a 1024px + workers  
        augmentations=transformar\_albumentations\_vbt(),  \# Inyección nativa vía kwarg (data/augment.py:2834); NO callback  
        seed=42,  
        project=str(dir\_base / "runs" / "detect"),  
        name="train\_vbt\_rocm\_1024",  
        exist\_ok=True,  
        save=True,  
        save\_period=10,  
        plots=True,  
    )  
      
    \# Evaluación rigurosa sobre el conjunto TEST (nunca visto en train ni val)  
    ruta\_best \= Path(resultados.save\_dir) / "weights" / "best.pt"  
    modelo\_final \= YOLO(str(ruta\_best))  
      
    metricas\_test \= modelo\_final.val(  
        data=str(ruta\_data),  
        split="test",  
        imgsz=1024,  
        batch=8,  
        device=0,  
        plots=True,  
    )  
      
    precision \= float(metricas\_test.box.mp)  
    recall \= float(metricas\_test.box.mr)  
    f1\_max \= (2 \* precision \* recall) / max(precision \+ recall, 1e-6)  
    map50 \= float(metricas\_test.box.map50)  
    map50\_95 \= float(metricas\_test.box.map)  
      
    reporte\_metricas \= {  
        "modelo\_ruta": str(ruta\_best),  
        "precision": round(precision, 4),  
        "recall": round(recall, 4),  
        "f1\_score": round(f1\_max, 4),  
        "mAP50": round(map50, 4),  
        "mAP50\_95": round(map50\_95, 4),  
    }  
      
    with open(dir\_base / "metricas\_produccion.json", "w", encoding="utf-8") as f:  
        json.dump(reporte\_metricas, f, indent=4)  
          
    logger.info(f"RESULTADOS TEST \-\> F1: {f1\_max:.4f} | mAP50: {map50:.4f} | mAP50-95: {map50\_95:.4f}")

if \_\_name\_\_ \== "\_\_main\_\_":  
    entrenar\_y\_evaluar()

\`\`\`

\#\#\# 5.3 Protocolo de Verificación CLI (Gate 5\)

\`\`\`bash  
python3 "Dataset\_YOLOv8/Entrenar.py"  
\# Monitoreo en otra terminal durante el entrenamiento para verificar saturación de GPU AMD:  
rocm-smi \--showuse \--showmeminfo vram  
\# Criterio de aprobación en Dataset\_YOLOv8/metricas\_produccion.json:  
\# F1-Score \>= 0.94 | mAP50 \>= 0.96 | mAP50-95 \>= 0.82

\`\`\`

\---

\#\# Fase 6: Reingeniería del Seguimiento en \`Creardataset.py\` (BoT-SORT con Compensación de Movimiento de Cámara)

\#\#\# 6.1 Explicación Técnica

El código actual de \`Creardataset.py\` invoca \`tracker="bytetrack.yaml"\` con \`conf=0.15\`. ¿Por qué falla cuando cae el peso o vibra la cámara?

\* \*\*Limitación del Filtro de Kalman en ByteTrack:\*\* ByteTrack asume que la cámara está estática. Cuando el atleta suelta 200+ kg en el suelo, la onda de choque sacude el teléfono. En un solo frame (\$33\\text{ ms}\$), todo el cuadro da un salto brusco de \$40\\text{ a }80\\text{ píxeles}\$. El filtro de Kalman de ByteTrack proyecta la caja del disco siguiendo la inercia previa, calcula el \`IoU\` contra la nueva posición desplazada por el temblor de cámara, obtiene un \`IoU\` cercano a cero y \*\*destruye o cambia el Track ID\*\*.  
\* \*\*Solución con BoT-SORT \+ GMC (\`sparseOptFlow\`):\*\* BoT-SORT extrae puntos característicos de las esquinas del fondo del gimnasio en cada frame y calcula el \*\*Flujo Óptico Disperso de Lucas-Kanade\*\* con rechazo de \*outliers\* (RANSAC). Con ello obtiene una matriz de homografía/afín que mide exactamente cuántos píxeles se movió la cámara y \*\*corrige las coordenadas del Filtro de Kalman antes de medir el IoU con el disco\*\*.  
\* \*\*Asociación de Dos Etapas Ajustada para Blur:\*\* Si durante el impacto el \*motion blur\* hace que la confianza de YOLO baje de \`0.75\` a \`0.14\`, la segunda etapa de asociación de BoT-SORT (\`track\_low\_thresh: 0.08\`) rescata la detección usando proximidad espacial corregida por cámara, manteniendo vivo el \`id\_fijado\` sin interrumpir la curva de velocidad de \`EstimadorCinematica\`.

\#\#\# 6.2 Directivas de Código para el Agente (\`config/vbt\_botsort.yaml\` y actualización de \`Creardataset.py\`)

1\. \*\*Crear el archivo \`config/vbt\_botsort.yaml\` en la raíz del servicio:\*\*  
\`\`\`yaml  
\# Configuración BoT-SORT especializada para VBT (Resiliencia a caída de barra y vibración)  
tracker\_type: botsort  
track\_high\_thresh: 0.42   \# Primera asociación: detecciones claras  
track\_low\_thresh: 0.08    \# Segunda asociación: rescata discos con Motion Blur severo en caídas/tirones  
new\_track\_thresh: 0.50    \# Evita inicializar tracks falsos en objetos del fondo  
track\_buffer: 60          \# Mantiene la memoria cinemática hasta 2.0s (a 30fps) o 1.0s (a 60fps)  
match\_thresh: 0.80        \# Tolerancia de emparejamiento tras compensar vibración de cámara  
fuse\_score: True          \# Pondera IoU con score de confianza

\# Compensación de Movimiento Global de Cámara (GMC)  
gmc\_method: sparseOptFlow \# Flujo óptico Lucas-Kanade para absorber el golpe de los discos contra el suelo

\# Re-ID Apariencia desactivado (todos los discos son idénticos; activar ReID añade latencia sin beneficio)  
proximity\_thresh: 0.5  
appearance\_thresh: 0.25  
with\_reid: False

\`\`\`

2\. \*\*Modificaciones Quirúrgicas en \`Creardataset.py\` (Manteniendo 100% compatibilidad de API):\*\*  
\* En \`ConfigVBT\`:  
\* Añadir \`ruta\_tracker: str \= "config/vbt\_botsort.yaml"\` (con fallback automático a \`"botsort.yaml"\` si no encuentra la ruta personalizada).  
\* Ajustar \`conf\_deteccion: float \= 0.08\` (para permitir que \`track\_low\_thresh\` de BoT-SORT reciba los cuadros con \*motion blur\* extremo; el propio tracker descartará el ruido que no coincida con el filtro de Kalman del disco activo).

\* En \`RastreadorDisco.\_seleccionar\_caja\`:  
\* Agregar \*\*Predicción Cinemática Lineal en el Re-enganche\*\*: Actualmente, cuando se pierde el ID por unos frames, \`RastreadorDisco\` busca cerca de \`self.ultima\_pos\` estática. Modifícalo para guardar \`self.ultima\_vel\_px \= (vx\_px, vy\_px)\` y proyectar la posición esperada \`pos\_esperada \= self.ultima\_pos \+ self.ultima\_vel\_px \* min(self.frames\_sin\_deteccion, 8)\` antes de medir la distancia euclidiana \`dists\`. Si la barra iba subiendo rápido cuando ocurrió la oclusión por las manos/rodillas, buscará arriba de la última posición, no atrás.

\---

\#\# Fase 7: Exportación Multi-Target y Despliegue (Servidor ROCm \+ Móvil NCNN)

\#\#\# 7.1 Explicación Técnica

El ecosistema requiere alimentar dos entornos de ejecución sin degradar el F1-Score:

1\. \*\*Modo Post-Procesado y Tiempo Real en Servidor (WhatsApp Bot / Backend FastAPI):\*\*  
En lugar de re-instanciar el modelo en cada llamada de \`procesar\_y\_responder\_video\_vbt\`, o sufrir sobrecarga de inicialización, el checkpoint ganador (\`best.pt\`) se copia a \`models/best.pt\` y se valida su inferencia en FP16 bajo ROCm. Adicionalmente, se exporta a \*\*ONNX (\`opset=17\`, \`simplify=True\`)\*\* para interoperabilidad.  
2\. \*\*Modo Móvil (\`Entrenar\_Mobile.py\`):\*\*  
Se exporta a \*\*NCNN\*\* (\`best\_ncnn\_model/\`) ejecutando un test comparativo automático que mide la desviación entre las cajas predichas por \`best.pt\` (FP16 en GPU) y \`best\_ncnn\_model\` sobre el conjunto \`test\`.

\#\#\# 7.2 Directivas de Código para el Agente (\`Dataset\_YOLOv8/Exportar\_Produccion.py\`)

Crea el script \`Exportar\_Produccion.py\` que realice:

1\. Carga de \`runs/detect/train\_vbt\_rocm\_1024/weights/best.pt\`.  
2\. Exportación a ONNX (\`model.export(format="onnx", imgsz=1024, half=False, simplify=True, opset=17)\`).  
3\. Exportación a NCNN (\`model.export(format="ncnn", imgsz=640, half=True)\` para versión móvil).  
4\. \*\*Benchmark de Paridad y Latencia:\*\*  
\* Corre 100 iteraciones de calentamiento (\*warmup\*) y 200 de medición en la GPU ROCm a \`imgsz=1024\` en FP16.  
\* Verifica que el throughput en tiempo real supere los \*\*45 FPS\*\* a \`1024x1024\`.  
\* Si pasa las pruebas, realiza un despliegue atómico copiando \`best.pt\` hacia la carpeta \`models/best.pt\` que consume \`vbt\_service.py\`.

\---

\#\# Fase 8: Arnés de Validación End-to-End contra Videos Reales (\`test\_e2e\_vbt.py\`)

\#\#\# 8.1 Explicación Técnica

Las métricas estáticas por frame (\`mAP\`, \`F1\`) no garantizan por sí solas que el motor cinemático completo (\`Creardataset.py\`) cuente las repeticiones y mida las velocidades sin errores. El paso final obligatorio antes de dar por cerrado el ciclo MLOps es ejecutar una \*\*Suite de Regresión Cinemática sobre Videos de Validación\*\*.

Se seleccionan al menos \*\*6 videos reales de control\*\* (2 de Sentadilla, 2 de Press de Banca, 2 de Peso Muerto con caída rápida de barra/vibración de cámara) con su \*Ground Truth\* anotado en un JSON (\`reps\_reales\`, \`ejercicio\`, \`carga\_kg\`).

\#\#\# 8.2 Directivas de Código para el Agente (\`tests/test\_e2e\_vbt.py\`)

Construye un script de validación automatizada que importe \`procesar\_levantamiento\_vbt\` desde \`services.Creardataset\` (anulando con un \*mock\* la llamada a \`enviar\_mensaje\_whatsapp\` para no enviar mensajes reales durante el test) y evalúe cada video calculando los siguientes \*\*KPIs de Aceptación en Producción\*\*:

| Métrica de Validación en Video | Fórmula / Fuente en \`Creardataset.py\` | Umbral Mínimo de Aprobación |  
| \--- | \--- | \--- |  
| \*\*Cobertura de Tracking (%)\*\* | \`100 \* frames\_con\_disco / frames\_totales\` | \`\>= 95.0%\` en Sentadilla/Banca, \`\>= 92.0%\` en Peso Muerto |  
| \*\*Exactitud de Conteo de Reps\*\* | \`abs(resultado\["reps"\] \- ground\_truth\["reps"\])\` | Error exacto \`== 0\` repeticiones |  
| \*\*Continuidad Cinemática\*\* | \`estimador.reinicios\_por\_hueco\` | \`== 0\` reinicios de ventana durante fases concéntricas |  
| \*\*Estabilidad de Escala (\`px-\>m\`)\*\* | Coeficiente de variación de \`det.alto\_caja\` tras semilla | \`CV \<= 4.0%\` a lo largo de toda la serie |  
| \*\*Falsos Positivos de Velocidad\*\* | \`max(r\["v\_peak"\] for r in detalle\_repeticiones)\` | \`\<= 2.2 m/s\` (cero picos espurios por salto de caja) |  
| \*\*Velocidad de Procesado (FPS)\*\* | \`frames\_totales / tiempo\_ejecucion\_s\` | \`\>= 30 FPS\` (incluyendo renderizado HUD y pipe FFmpeg) |

\#\#\# 8.3 Secuencia de Ejecución Completa para el Agente CLI

Para ejecutar el flujo de principio a fin y validar cada etapa de forma determinista, sigue esta secuencia exacta de comandos:

\`\`\`bash  
\# 1\. Extraer frames inteligentes con captura de Motion Blur e identificadores de video  
python3 "Videos para entrenamiento/extraer\_frames.py" \--max-frames 45 \--workers 8

\# 2. Auditar etiquetas actuales y saneamiento
python3 "Dataset_YOLOv8/auditar_dataset.py" --fix-classes --report  

# 2.1 Etiquetado manual asistido de fotogramas (GUI PySide6 con Zoom, 8 tiradores de ajuste y pre-anotación IA)
python3 etiquetar.py                  # Fotogramas pendientes en images_sin_etiquetar
python3 etiquetar.py --cola           # Revisión humana prioritaria (cola_revision_humana.txt)
python3 etiquetar.py --video "Videos para entrenamiento/20261019_105121.mp4" # Captura y etiquetado directo desde video


\# 3\. Partición 70/20/10 agrupada por video\_id (Cero Data Leakage) \+ negativos de fondo  
python3 "Dataset\_YOLOv8/Dividir.py" \--train 0.70 \--val 0.20 \--test 0.10 \--verify-no-leakage

\# 4\. Entrenamiento exhaustivo sobre ROCm (1024px, Albumentations Blur, AdamW, cos\_lr, close\_mosaic)  
python3 "Dataset\_YOLOv8/Entrenar.py"

\# 5\. Exportación a producción (Actualización de models/best.pt, ONNX y NCNN Mobile)  
python3 "Dataset\_YOLOv8/Exportar\_Produccion.py" \--deploy

\# 6\. Prueba de fuego End-to-End sobre videos de validación reales con el motor VBT \+ BoT-SORT GMC  
python3 "tests/test\_e2e\_vbt.py" \--videos-dir "Videos para entrenamiento/validacion\_e2e" \--report "reporte\_e2e\_vbt.json"

\`\`\`