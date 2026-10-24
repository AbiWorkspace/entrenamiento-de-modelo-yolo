"""Test E2E del motor VBT (Fase 8).

Verifica la estructura del dict de retorno de ``procesar_levantamiento_vbt``
y los KPIs de robustez de tracking (cobertura_frames, reinicios_por_hueco)
sobre videos reales de ``Videos para entrenamiento/``.

Uso:
    pytest tests/test_e2e_vbt.py -v
    python tests/test_e2e_vbt.py --max-videos 6 --report reporte_e2e_vbt.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType

logger = logging.getLogger("test_e2e_vbt")

# ------------------------------------------------------------------ #
# Bootstrap: stub de data.meta_service + import del motor desde ROOT #
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_data_meta_service() -> None:
    """Inyecta un módulo fake ``data.meta_service`` antes de importar Creardataset."""
    if "data.meta_service" in sys.modules:
        return

    def enviar_mensaje_whatsapp(
        texto: str = "", telefono_destino: str = "", **kwargs
    ) -> bool:  # noqa: D401
        logger.info(
            "WhatsApp (stub): %s -> %s (%d chars)",
            telefono_destino,
            texto[:60],
            len(texto),
        )
        return True

    data_pkg = ModuleType("data")
    meta_service = ModuleType("data.meta_service")
    setattr(meta_service, "enviar_mensaje_whatsapp", enviar_mensaje_whatsapp)
    data_pkg.meta_service = meta_service  # type: ignore[attr-defined]
    sys.modules["data"] = data_pkg
    sys.modules["data.meta_service"] = meta_service


_stub_data_meta_service()

import Creardataset  # noqa: E402

# ------------------------------------------------------------------ #
# Constantes y configuración                                          #
# ------------------------------------------------------------------ #
VIDEOS_DIR = ROOT / "Videos para entrenamiento"
MODEL_PATH = ROOT / "Dataset_YOLOv8" / "best.pt"
OUT_DIR = Path("/tmp/opencode/e2e_vbt")

EJERCICIOS_CICLO: list[str] = ["Sentadilla", "Press de Banca", "Peso Muerto"]

CLAVES_ESPERADAS: tuple[str, ...] = (
    "estado",
    "reps",
    "v_promedio_global",
    "detalle_repeticiones",
    "ruta_video_renderizado",
    "analisis_serie",
    "reporte_whatsapp",
    "reporte_tecnico",
    "carga_kg",
    "e1rm_estimado_kg",
    "vl_pct",
    "reps_en_reserva",
    "cobertura_frames",
    "reinicios_por_hueco",
)

# Clave presente en el retorno del motor pero omitida intencionalmente en el reporte JSON.
CLAVE_ADMITIDA_EXTRA = "contexto_agente"

MAX_V_PEAK_M_S = 2.2  # tope físico razonable para velocidad de disco


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _seleccionar_videos(max_videos: int) -> list[tuple[Path, str]]:
    """Devuelve hasta ``max_videos`` tuples (ruta_video, nombre_ejercicio)."""
    if not VIDEOS_DIR.is_dir():
        raise FileNotFoundError(f"Directorio de videos no existe: {VIDEOS_DIR}")

    videos = sorted(
        (p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() == ".mp4" and p.is_file()),
        key=lambda p: p.name,
    )
    if not videos:
        raise FileNotFoundError(f"No hay .mp4 en {VIDEOS_DIR}")

    seleccion = videos[:max_videos]
    return [
        (v, EJERCICIOS_CICLO[i % len(EJERCICIOS_CICLO)])
        for i, v in enumerate(seleccion)
    ]


def _validar_resultado(resultado: dict) -> None:
    """Aserciones sobre estructura y rangos de los KPIs (Plan §8.2)."""
    faltantes = [k for k in CLAVES_ESPERADAS if k not in resultado]
    assert not faltantes, f"Faltan claves en retorno: {faltantes}"

    assert isinstance(resultado["reps"], int), "reps debe ser int"
    assert resultado["reps"] >= 0, f"reps negativo: {resultado['reps']}"

    v_prom = float(resultado["v_promedio_global"])
    assert v_prom >= 0.0, f"v_promedio_global negativa: {v_prom}"

    cov = float(resultado["cobertura_frames"])
    assert 0.0 <= cov <= 1.0, f"cobertura_frames fuera de [0,1]: {cov}"

    rein = resultado["reinicios_por_hueco"]
    assert isinstance(rein, int) and rein >= 0, (
        f"reinicios_por_hueco inválido: {rein!r}"
    )

    detalle = resultado["detalle_repeticiones"]
    assert isinstance(detalle, list), "detalle_repeticiones debe ser lista"

    v_peaks = [float(r.get("v_peak", 0.0)) for r in detalle if isinstance(r, dict)]
    if v_peaks:
        max_peak = max(v_peaks)
        assert max_peak <= MAX_V_PEAK_M_S, (
            f"v_peak máxima {max_peak:.3f} m/s excede tope físico {MAX_V_PEAK_M_S} m/s"
        )

    if resultado["estado"] == "exito":
        renderizado = Path(resultado["ruta_video_renderizado"])
        assert renderizado.is_file(), f"Video renderizado no existe: {renderizado}"
        assert renderizado.stat().st_size > 0, "Video renderizado vacío"


def _procesar_video(
    ruta_video: Path,
    nombre_ejercicio: str,
    index: int,
    ruta_modelo: Path = MODEL_PATH,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = OUT_DIR / f"e2e_{index}_{ruta_video.stem}.mp4"

    t0 = time.perf_counter()
    logger.info("[E2E %d] %s (%s)", index, ruta_video.name, nombre_ejercicio)
    resultado = Creardataset.procesar_levantamiento_vbt(
        ruta_modelo=str(ruta_modelo),
        ruta_video=str(ruta_video),
        media_id=f"e2e_{index}",
        numero_usuario="0000000000",
        nombre_ejercicio=nombre_ejercicio,
        ruta_salida=str(ruta_salida),
        carga_kg=None,
        nivel_atleta="Avanzado",
        ruta_modelo_pose=None,
    )
    dur = time.perf_counter() - t0
    logger.info(
        "[E2E %d] estado=%s reps=%s en %.1fs",
        index,
        resultado.get("estado"),
        resultado.get("reps"),
        dur,
    )
    return {
        "video": ruta_video.name,
        "ejercicio": nombre_ejercicio,
        "segundos": round(dur, 2),
        **{k: v for k, v in resultado.items() if k != "contexto_agente"},
    }


# ------------------------------------------------------------------ #
# Pytest                                                             #
# ------------------------------------------------------------------ #
def test_modelo_existe() -> None:
    assert MODEL_PATH.is_file(), f"Modelo no encontrado: {MODEL_PATH}"


def test_e2e_video_corto() -> None:
    """Procesa el primer video (el más corto por orden alfabético) y valida KPIs."""
    (ruta_video, ejercicio) = _seleccionar_videos(1)[0]
    res = _procesar_video(ruta_video, ejercicio, 0)
    assert res.get("estado") in ("exito", "error"), (
        f"estado inesperado: {res.get('estado')}"
    )
    if res.get("estado") == "exito":
        _validar_resultado(res)


def test_e2e_bateria_videos(max_videos: int = 6) -> None:
    """Batería completa: hasta ``max_videos`` videos con labels cíclicos."""
    seleccion = _seleccionar_videos(max_videos)
    for i, (ruta_video, ejercicio) in enumerate(seleccion):
        res = _procesar_video(ruta_video, ejercicio, i + 1)
        if res.get("estado") == "exito":
            _validar_resultado(res)


# ------------------------------------------------------------------ #
# Runner standalone (sin pytest)                                     #
# ------------------------------------------------------------------ #
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="E2E VBT standalone")
    parser.add_argument("--max-videos", type=int, default=6)
    parser.add_argument(
        "--report", type=str, default=str(ROOT / "reporte_e2e_vbt.json")
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Ruta al .pt (default: Dataset_YOLOv8/best.pt)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not Path(args.model).is_file():
        logger.error("Modelo no existe: %s", args.model)
        return 1

    resultados: list[dict] = []
    fallos = 0
    for i, (ruta_video, ejercicio) in enumerate(_seleccionar_videos(args.max_videos)):
        try:
            res = _procesar_video(ruta_video, ejercicio, i + 1, Path(args.model))
            if res.get("estado") == "exito":
                _validar_resultado(res)
                res["kpi_ok"] = True
            else:
                res["kpi_ok"] = None  # video sin reps válidas: se reporta pero no falla
                logger.warning(
                    "[E2E %d] estado=%s (mensaje: %s)",
                    i + 1,
                    res.get("estado"),
                    str(res.get("mensaje", ""))[:120],
                )
        except Exception as e:
            fallos += 1
            res = {
                "video": ruta_video.name,
                "ejercicio": ejercicio,
                "error": f"{type(e).__name__}: {e}",
                "kpi_ok": False,
            }
            logger.exception("[E2E %d] excepción", i + 1)
        resultados.append(res)

    reporte = {
        "model": str(args.model),
        "total_videos": len(resultados),
        "fallos": fallos,
        "resultados": resultados,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Reporte E2E guardado en %s", out)

    print(
        json.dumps(
            {"total": len(resultados), "fallos": fallos, "reporte": str(out)},
            ensure_ascii=False,
        )
    )
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
