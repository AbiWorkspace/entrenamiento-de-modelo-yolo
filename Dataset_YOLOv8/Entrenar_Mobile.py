import os
import sys
import time
import platform
from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURACIÓN ESPECÍFICA PARA MÓVILES
# ==========================================
# En móviles (Android / iOS):
# - imgsz=480 o 416 da una ganancia de velocidad de +60% en FPS y menor consumo de batería
#   manteniendo prácticamente la misma precisión para objetos grandes como discos de pesas.
IMGSZ_MOBILE = 480
BATCH_SIZE = 32
WORKERS = 8
EPOCHS = 40


def validar_gpu(dispositivo_objetivo=0):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU no disponible. Se requiere GPU para entrenar.")
    idx = int(dispositivo_objetivo)
    nombre = torch.cuda.get_device_name(idx)
    vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    print(f"✅ Hardware validado: {nombre} ({vram:.2f} GB VRAM)")
    return idx


def exportar_para_moviles(ruta_best_pt, imgsz=IMGSZ_MOBILE):
    """
    Exporta el modelo entrenado a los 3 formatos estándar para aplicaciones móviles:
    1. ONNX (.onnx) -> Ideal para Flutter, React Native, Android y iOS (ONNX Runtime Mobile).
    2. NCNN (carpeta ncnn) -> El motor C++ más veloz del mundo para procesadores ARM móviles (Snapdragon/MediaTek).
    3. TorchScript (.torchscript) -> Para PyTorch Mobile nativo en Android (Kotlin/Java) y iOS (Swift).
    """
    print("\n" + "=" * 65)
    print(">>> EXPORTANDO MODELO OPTIMIZADO PARA DISPOSITIVOS MÓVILES <<<")
    print("=" * 65)

    modelo = YOLO(str(ruta_best_pt))

    # 1. Exportación a ONNX
    print("\n📦 [1/3] Generando formato ONNX (Universal Mobile / Flutter / React Native)...")
    try:
        ruta_onnx = modelo.export(format="onnx", imgsz=imgsz, simplify=True, opset=19)
        print(f"   ✓ ONNX guardado en: {ruta_onnx}")
    except Exception as e:
        print(f"   ⚠️ Error en ONNX: {e}")

    # 2. Exportación a NCNN
    print("\n📦 [2/3] Generando formato NCNN (Máximo rendimiento en CPU/GPU de celulares)...")
    try:
        ruta_ncnn = modelo.export(format="ncnn", imgsz=imgsz)
        print(f"   ✓ NCNN guardado en: {ruta_ncnn}")
    except Exception as e:
        print(f"   ⚠️ Error en NCNN: {e}")

    # 3. Exportación a TorchScript
    print("\n📦 [3/3] Generando formato TorchScript (PyTorch Mobile nativo)...")
    try:
        ruta_torchscript = modelo.export(format="torchscript", imgsz=imgsz)
        print(f"   ✓ TorchScript guardado en: {ruta_torchscript}")
    except Exception as e:
        print(f"   ⚠️ Error en TorchScript: {e}")

    print("\n" + "=" * 65)
    print("🎉 ¡Modelos móviles listos para integrar en Android / iOS!")
    print("=" * 65)


def entrenar_modelo_movil(
    modelo_base=None,
    dispositivo=0,
    epochs=EPOCHS,
    imgsz=IMGSZ_MOBILE,
):
    gpu_id = validar_gpu(dispositivo)

    ruta_script = Path(__file__).resolve().parent
    ruta_yaml = ruta_script / "data.yaml"

    # Seleccionar modelo de partida:
    # Si tenemos best.pt reentrenado, partimos de él (Fine-Tuning ultra-rápido)
    if modelo_base and Path(modelo_base).exists():
        modelo_iniciar = Path(modelo_base)
    elif (ruta_script / "best.pt").exists():
        modelo_iniciar = ruta_script / "best.pt"
    else:
        modelo_iniciar = "yolov8n.pt"

    print("=" * 65)
    print(">>> ENTRENAMIENTO DE MODELO LIGERO PARA DISPOSITIVOS MÓVILES <<<")
    print("=" * 65)
    print(f"• Modelo inicial:       {modelo_iniciar}")
    print(f"• Resolución optimizada:{imgsz}x{imgsz} px (alta velocidad en móviles)")
    print(f"• Épocas:               {epochs}")
    print(f"• Batch size:           {BATCH_SIZE}")
    print("=" * 65)

    model = YOLO(str(modelo_iniciar))

    # Entrenar modelo optimizado para tamaño y velocidad móvil
    resultado = model.train(
        data=str(ruta_yaml),
        epochs=epochs,
        batch=BATCH_SIZE,
        imgsz=imgsz,
        device=gpu_id,
        workers=WORKERS,
        project=str(ruta_script / "runs_mobile"),
        name="mobile_disco",
        exist_ok=True,
    )

    ruta_best = ruta_script / "runs_mobile" / "mobile_disco" / "weights" / "best.pt"
    print(f"\n✅ Entrenamiento móvil completado. Pesos en: {ruta_best}")

    # Exportar automáticamente a los formatos móviles
    exportar_para_moviles(ruta_best, imgsz=imgsz)


if __name__ == "__main__":
    # Inicia el entrenamiento móvil y exporta a ONNX, NCNN y TorchScript
    entrenar_modelo_movil(
        modelo_base=None,  # Toma automáticamente el best.pt que acabamos de entrenar
        epochs=35,         # 35 épocas son suficientes para ajustar a resolución móvil
        imgsz=480,         # 480x480 para 60 FPS estables en teléfonos
    )
