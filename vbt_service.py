import asyncio
import os
import traceback

import httpx

from core.config import settings
from data.meta_service import enviar_mensaje_whatsapp
from services.Creardataset import procesar_levantamiento_vbt

TOKEN_META = settings.ACCESS_TOKEN
ID_NUMERO_TELEFONO = settings.PHONE_NUMBER_ID

# Meta necesita unos segundos para encriptar el MP4 recién subido antes de poder
# reenviarlo en un mensaje.
ESPERA_PROCESADO_META = 2


async def _cliente() -> httpx.AsyncClient:
    """Reutiliza el pool HTTP del webhook: subir un video de 20 MB abriendo una
    conexión TLS nueva cada vez añade el handshake al tiempo total."""
    from api.rutas_whatsapp import cliente_http
    return await cliente_http()


async def _subir_y_enviar_video(cliente: httpx.AsyncClient, headers: dict,
                                numero_usuario: str, ruta_video: str,
                                caption: str) -> bool:
    """Sube el MP4 a Meta y lo manda al atleta. True si Meta lo aceptó."""
    url_upload = f"{settings.GRAPH_URL}/{ID_NUMERO_TELEFONO}/media"

    MAX_BYTES_META = int(15.0 * 1024 * 1024)
    if os.path.exists(ruta_video) and os.path.getsize(ruta_video) > MAX_BYTES_META:
        print(f"[RE-COMPRESION WHATSAPP]: El video pesa {os.path.getsize(ruta_video)/(1024*1024):.2f} MB. Comprimiendo para Meta (<15 MB)...")
        import subprocess
        import imageio_ffmpeg
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"
        ruta_comp = ruta_video + ".wa.mp4"
        cmd = [
            ffmpeg_exe, "-y", "-i", ruta_video,
            "-c:v", "libx264", "-crf", "25", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", ruta_comp
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(ruta_comp) and os.path.getsize(ruta_comp) <= MAX_BYTES_META:
            os.replace(ruta_comp, ruta_video)
        elif os.path.exists(ruta_comp):
            os.remove(ruta_comp)

    with open(ruta_video, "rb") as f_video:
        archivos = {"file": (os.path.basename(ruta_video), f_video, "video/mp4")}
        datos = {"messaging_product": "whatsapp"}
        respuesta_upload = await cliente.post(url_upload, headers=headers,
                                              data=datos, files=archivos,
                                              timeout=180.0)

    if respuesta_upload.status_code != 200:
        print(f"[ERROR UPLOAD]: {respuesta_upload.text[:400]}")
        return False

    nuevo_media_id = respuesta_upload.json().get("id")
    if not nuevo_media_id:
        print(f"[ERROR UPLOAD]: Meta no devolvió media_id -> {respuesta_upload.text[:300]}")
        return False

    await asyncio.sleep(ESPERA_PROCESADO_META)

    url_mensajes = f"{settings.GRAPH_URL}/{ID_NUMERO_TELEFONO}/messages"
    payload_mensaje = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": numero_usuario,
        "type": "video",
        "video": {"id": nuevo_media_id, "caption": caption},
    }
    respuesta_envio = await cliente.post(url_mensajes, headers=headers,
                                         json=payload_mensaje, timeout=60.0)

    if respuesta_envio.status_code in (200, 201):
        print("[EXITO META]: Video entregado exitosamente a la API de WhatsApp.")
        return True

    print("[ERROR META VIDEO]: Meta rechazo el envio del video.")
    print(f"[DETALLE EXACTO]: {respuesta_envio.text[:400]}")
    return False


async def procesar_y_responder_video_vbt(media_id_entrada: str, numero_usuario: str,
                                         nombre_ejercicio: str = "Movimiento Libre",
                                         carga_kg: float = None) -> str:
    """
    Orquestador completo: Descarga de Meta -> Procesa VBT -> Sube a Meta -> Envia WhatsApp.

    CONTRATO DE RETORNO (lo consume `procesar_y_enviar_mensaje`):
      · Éxito  -> el contexto VBT para el LLM (telemetría real de la serie).
      · Fallo  -> un string que EMPIEZA por "[ERROR]".

    Antes el camino de error devolvía "Error manejado con video forense enviado.",
    que no contenía "[ERROR]": el orquestador lo tomaba por telemetría legítima y
    se la pasaba al coach como si fueran las métricas de la serie.
    """
    ruta_script_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz_proyecto = os.path.dirname(ruta_script_actual)

    carpeta_data = os.path.join(ruta_raiz_proyecto, "data")
    carpeta_models = os.path.join(ruta_raiz_proyecto, "models")
    os.makedirs(carpeta_data, exist_ok=True)
    os.makedirs(carpeta_models, exist_ok=True)

    ruta_video_original = os.path.join(carpeta_data, f"input_{media_id_entrada}.mp4")
    ruta_video_procesado = os.path.join(carpeta_data, f"output_{media_id_entrada}.mp4")
    ruta_modelo = os.path.join(carpeta_models, "best.pt")

    headers_auth = {"Authorization": f"Bearer {TOKEN_META}"}

    try:
        cliente = await _cliente()

        print("[HTTP]: Solicitando URL del video a Meta...")
        respuesta_info = await cliente.get(f"{settings.GRAPH_URL}/{media_id_entrada}",
                                           headers=headers_auth)
        if respuesta_info.status_code != 200:
            print(f"[ERROR META]: No se pudo obtener la info del video. {respuesta_info.text[:300]}")
            await asyncio.to_thread(
                enviar_mensaje_whatsapp, numero_usuario,
                "No pude descargar tu video bro 🙏 Intenta enviarlo de nuevo.")
            return "[ERROR]: No pude descargar tu video."

        url_descarga = respuesta_info.json().get("url")
        if not url_descarga:
            await asyncio.to_thread(
                enviar_mensaje_whatsapp, numero_usuario,
                "Meta no me dejó bajar tu video bro 🙏 Reenvíalo porfa.")
            return "[ERROR]: Meta no devolvió URL de descarga."

        print("[HTTP]: Descargando archivo fisico MP4...")
        respuesta_video = await cliente.get(url_descarga, headers=headers_auth, timeout=120.0)
        respuesta_video.raise_for_status()
        with open(ruta_video_original, "wb") as f:
            f.write(respuesta_video.content)

        print("[SISTEMA]: Entregando video al motor VBT local...")
        # YOLO + OpenCV + moviepy sobre el video completo tarda MINUTOS. Ejecutarlo
        # directamente en la corrutina congelaba el servidor entero durante todo ese
        # tiempo (ni webhooks de otros atletas, ni health check). Va a un hilo.
        resultado_vbt = await asyncio.to_thread(
            procesar_levantamiento_vbt,
            ruta_modelo=ruta_modelo,
            ruta_video=ruta_video_original,
            ruta_salida=ruta_video_procesado,
            media_id=media_id_entrada,
            numero_usuario=numero_usuario,
            nombre_ejercicio=nombre_ejercicio,
            carga_kg=carga_kg,          # <--- AQUÍ VIAJA EL PESO PARA EL 1RM
        )

        if not isinstance(resultado_vbt, dict):
            return "[ERROR]: El motor VBT no devolvió un resultado utilizable."

        # `.get`, no `[...]`: un resultado sin la clave 'estado' reventaba con
        # KeyError y el atleta se quedaba sin respuesta ni explicación.
        if resultado_vbt.get("estado") == "error":
            mensaje_error = resultado_vbt.get("mensaje", "Hubo un error en el análisis de video.")
            # enviar_mensaje_whatsapp es síncrona (requests + pausas): a un hilo.
            await asyncio.to_thread(enviar_mensaje_whatsapp, numero_usuario,
                                    f"*Aviso VBT:*\n{mensaje_error}")

            ruta_forense = resultado_vbt.get("ruta_video_renderizado")
            if ruta_forense and os.path.exists(ruta_forense):
                print("[HTTP]: Subiendo video forense a Meta...")
                try:
                    await _subir_y_enviar_video(cliente, headers_auth, numero_usuario,
                                                ruta_forense, "Video de diagnóstico VBT")
                except Exception as e_forense:
                    print(f"[ERROR UPLOAD FORENSE]: {e_forense}")
            return f"[ERROR]: {mensaje_error}"

        if not os.path.exists(ruta_video_procesado):
            return "[ERROR]: El análisis terminó pero no se generó el video de salida."

        print("[HTTP]: Subiendo video renderizado a los servidores de Meta...")
        enviado = await _subir_y_enviar_video(
            cliente, headers_auth, numero_usuario, ruta_video_procesado,
            "Análisis VBT completado. Revisando tus métricas...")

        if not enviado:
            # El análisis SÍ vale: se devuelve la telemetría igual y el coach
            # comenta los números aunque el video no haya podido viajar.
            print("[VBT]: el video no salió, pero la telemetría sigue siendo válida.")

        print("[EXITO]: Pipeline completado. Devolviendo contexto superalimentado para Qwen.")
        return resultado_vbt.get("contexto_agente", "Contexto VBT no disponible")

    except Exception:
        print("[CRASHEO PIPELINE]: Error inesperado en la integracion con Meta.")
        traceback.print_exc()
        return "[ERROR]: Ocurrio un error interno en el servidor al procesar el video."

    finally:
        for ruta in (ruta_video_original, ruta_video_procesado):
            if os.path.exists(ruta):
                try:
                    os.remove(ruta)
                except OSError as e:
                    print(f"[LIMPIEZA]: no se pudo borrar {ruta}: {e}")
        print("[LIMPIEZA]: Archivos temporales de video eliminados de data/.")
