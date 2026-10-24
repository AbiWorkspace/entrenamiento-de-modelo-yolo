"""Dividir.py — Fase 3 (Gate 3): partición anti-fuga por video.

Particiona SOLO el conjunto `vid_*` (frames extraídos en Fase 1) en
train/val/test a nivel de VIDEO para evitar fuga de información entre splits.
El legado `frame_*` se cuarentena en `_quarantine_legacy/`.

Algoritmo:
  1. Cuarentena de `frame_*` (idempotente).
  2. Recolecta pool de `vid_*` etiquetados desde images/{train,val,test}.
  3. Estratificación por video + indicador "suelo" (algún box con cy >= 0.68).
  4. Split 70/20/10 por estrato con largest-remainder, semilla fija.
  5. Distribuye archivos a images/<split> y labels/<split>.
  6. Genera data.yaml (ruta absoluta).
  7. `--verify-no-leakage`: aserta intersección vacía entre video_id de splits.

Uso:
    python Dividir.py --train 0.70 --val 0.20 --test 0.10 --seed 42
    python Dividir.py --verify-no-leakage
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("dividir")

BASE: Path = Path(__file__).resolve().parent
IMAGES: Path = BASE / "images"
LABELS: Path = BASE / "labels"
QUARANTINE: Path = BASE / "_quarantine_legacy"
DATA_YAML: Path = BASE / "data.yaml"

SPLITS: tuple[str, ...] = ("train", "val", "test")
VID_RE = re.compile(r"^(?P<vid>vid_.+)_f(?P<n>\d+)$")


@dataclass(frozen=True)
class RunConfig:
    train: float
    val: float
    test: float
    seed: int

    def ratios(self) -> tuple[float, ...]:
        return (self.train, self.val, self.test)


def video_id_of(image_path: Path) -> str | None:
    """Extrae el identificador de video desde el nombre `vid_<hash>_..._f<NNN>`."""
    m = VID_RE.match(image_path.stem)
    return m.group("vid") if m else None


def suelo_in_label(label_path: Path, cy_min: float = 0.68) -> bool:
    """Indicador: algún box con centro-y (campo 2) >= cy_min."""
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cy = float(parts[2])
                if cy >= cy_min:
                    return True
    except (OSError, ValueError):
        pass
    return False


def largest_remainder(ratios: tuple[float, ...], total: int) -> list[int]:
    """Distribuye `total` enteros según proporciones con resto-mayor."""
    raw = [r * total for r in ratios]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    if remainder <= 0:
        return floors
    # Ordena por fracción descendente y reparte el resto.
    fracs = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i], i), reverse=True)
    for i in fracs[:remainder]:
        floors[i] += 1
    return floors


def split_stratum(
    video_ids: list[str], ratios: tuple[float, ...], seed: int, name: str
) -> dict[str, list[str]]:
    """Divide una lista de video_ids en train/val/test de forma determinista."""
    ordered = sorted(video_ids)
    rng = random.Random(f"gate3-{seed}-{name}")
    rng.shuffle(ordered)
    counts = largest_remainder(ratios, len(ordered))
    out: dict[str, list[str]] = {}
    idx = 0
    for split, n in zip(SPLITS, counts):
        out[split] = ordered[idx : idx + n]
        idx += n
    return out


def quarantine_legacy() -> int:
    """Mueve `frame_*` desde images/<split> y labels/<split> a _quarantine_legacy/."""
    moved = 0
    for split in SPLITS:
        for kind, root in (("images", IMAGES), ("labels", LABELS)):
            src_dir = root / split
            dst_dir = QUARANTINE / kind / split
            if not src_dir.is_dir():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.iterdir()):
                if not f.is_file() or not f.name.startswith("frame_"):
                    continue
                shutil.move(str(f), str(dst_dir / f.name))
                moved += 1
    log.info("Cuarentena: %d archivos frame_* movidos", moved)
    return moved


def collect_pool() -> dict[str, list[tuple[Path, Path]]]:
    """Pool de vid_* etiquetados agrupado por video_id, desde los tres splits y staging/manual."""
    pool: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    vistos: set[str] = set()

    # 1. Recorrer los tres splits oficiales (train, val, test)
    for split in SPLITS:
        img_dir = IMAGES / split
        if not img_dir.is_dir():
            continue
        for f in sorted(img_dir.iterdir()):
            if not f.is_file() or not f.name.startswith("vid_"):
                continue
            vid = video_id_of(f)
            if vid is None:
                continue
            lbl = LABELS / split / (f.stem + ".txt")
            if not lbl.exists():
                log.warning("Falta label para %s — omitido", f.name)
                continue
            pool[vid].append((f, lbl))
            vistos.add(f.name)

    # 2. Recorrer imágenes etiquetadas manualmente pendientes en images_sin_etiquetar
    sin_etq_dir = BASE / "images_sin_etiquetar"
    if sin_etq_dir.is_dir():
        for f in sorted(sin_etq_dir.iterdir()):
            if not f.is_file() or not f.name.startswith("vid_") or f.name in vistos:
                continue
            vid = video_id_of(f)
            if vid is None:
                continue
            # Buscar label correspondiente (junto a la imagen, en labels/train, o en labels/)
            lbl_candidatas = [
                sin_etq_dir / (f.stem + ".txt"),
                LABELS / "train" / (f.stem + ".txt"),
                LABELS / (f.stem + ".txt"),
            ]
            encontrado = None
            for c in lbl_candidatas:
                if c.is_file():
                    encontrado = c
                    break
            if encontrado is not None:
                pool[vid].append((f, encontrado))
                vistos.add(f.name)

    return dict(pool)


def compute_strata(
    pool: dict[str, list[tuple[Path, Path]]], cy_min: float = 0.68
) -> dict[str, list[str]]:
    """Estratifica videos: has_suelo / no_suelo."""
    strata: dict[str, list[str]] = {"has_suelo": [], "no_suelo": []}
    for vid in sorted(pool):
        if any(suelo_in_label(lbl, cy_min) for _img, lbl in pool[vid]):
            strata["has_suelo"].append(vid)
        else:
            strata["no_suelo"].append(vid)
    return strata


def write_data_yaml() -> None:
    content = (
        "path: " + str(BASE).replace("\\", "\\\\") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: {0: disco}\n"
    )
    DATA_YAML.write_text(content)
    log.info("data.yaml escrito en %s", DATA_YAML)


def verify_no_leakage() -> bool:
    """Aserta que ningún video_id aparece en más de un split."""
    by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    for split in SPLITS:
        img_dir = IMAGES / split
        if not img_dir.is_dir():
            continue
        for f in img_dir.iterdir():
            vid = video_id_of(f) if f.name.startswith("vid_") else None
            if vid is not None:
                by_split[split].add(vid)
    ok = True
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            inter = by_split[a] & by_split[b]
            if inter:
                log.error("FUGA %s ∩ %s = %s", a, b, sorted(inter)[:10])
                ok = False
    sizes = {s: len(by_split[s]) for s in SPLITS}
    log.info("video_ids por split: %s", sizes)
    return ok


def run(cfg: RunConfig) -> int:
    quarantine_legacy()

    pool = collect_pool()
    n_videos = len(pool)
    n_frames = sum(len(v) for v in pool.values())
    log.info("Pool: %d videos, %d frames vid_*", n_videos, n_frames)
    if n_videos == 0:
        log.error("Sin videos en el pool. Nada que particionar.")
        return 1

    strata = compute_strata(pool)
    for name in ("has_suelo", "no_suelo"):
        log.info("Estrato %-9s: %d videos", name, len(strata[name]))

    assignment: dict[str, list[str]] = {s: [] for s in SPLITS}
    for name, vids in strata.items():
        if not vids:
            continue
        parts = split_stratum(vids, cfg.ratios(), cfg.seed, name)
        for s in SPLITS:
            assignment[s].extend(parts[s])

    # Distribuir archivos a su split destino.
    vid_to_split: dict[str, str] = {}
    for s in SPLITS:
        for v in assignment[s]:
            vid_to_split[v] = s

    for s in SPLITS:
        (IMAGES / s).mkdir(parents=True, exist_ok=True)
        (LABELS / s).mkdir(parents=True, exist_ok=True)

    moved = 0
    for vid, entries in pool.items():
        target = vid_to_split[vid]
        for img, lbl in entries:
            dst_img = IMAGES / target / img.name
            dst_lbl = LABELS / target / (img.stem + ".txt")
            if img.parent != dst_img.parent:
                shutil.move(str(img), str(dst_img))
                moved += 1
            if lbl.parent != dst_lbl.parent:
                shutil.move(str(lbl), str(dst_lbl))
                moved += 1
    log.info("Distribución: %d archivos movidos", moved)

    write_data_yaml()

    # Resumen.
    for s in SPLITS:
        n_imgs = len(list((IMAGES / s).glob("vid_*")) if (IMAGES / s).is_dir() else [])
        log.info("  %-5s: %d videos, %d frames", s, len(assignment[s]), n_imgs)

    ok = verify_no_leakage()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Partición anti-fuga por video (Gate 3)")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.20)
    p.add_argument("--test", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-no-leakage", action="store_true")
    args = p.parse_args(argv)

    if args.verify_no_leakage:
        return 0 if verify_no_leakage() else 1

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        log.error("Proporciones no suman a 1.0 (suman %.4f)", total)
        return 2
    cfg = RunConfig(args.train, args.val, args.test, args.seed)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
