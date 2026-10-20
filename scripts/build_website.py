#!/usr/bin/env python3
"""Merge PNG result archives downloaded from both Colab notebooks into the site."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def copy_pngs(archive: Path, destination: Path) -> list[str]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    copied: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cs180-results-") as temp:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                path = Path(name)
                if path.suffix.lower() != ".png" or path.is_absolute() or ".." in path.parts:
                    continue
                target = Path(temp) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
            for image in Path(temp).rglob("*.png"):
                dest = destination / image.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, dest)
                copied.append(image.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parta", type=Path, required=True, help="parta-website-results.zip downloaded from Colab")
    parser.add_argument("--partb", type=Path, required=True, help="partb-website-results.zip downloaded from Colab")
    parser.add_argument("--site", type=Path, default=ROOT, help="Website project directory (default: this project)")
    args = parser.parse_args()
    site = args.site.resolve()
    parta = copy_pngs(args.parta.resolve(), site / "assets" / "results" / "parta")
    partb = copy_pngs(args.partb.resolve(), site / "assets" / "results" / "partb")
    expected = {
        "parta": {"text_edit_campanile.png", "anagram_views_0.png", "hybrid_0.png", "conditional_samples.png"},
        "partb": {"denoiser_loss.png", "time_flow_loss.png", "time_flow_samples_epoch_10.png", "class_flow_loss.png", "class_flow_samples_epoch_10.png"},
    }
    missing = {part: sorted(names - set(parta if part == "parta" else partb)) for part, names in expected.items()}
    print(f"Copied {len(parta)} Part A images and {len(partb)} Part B images to {site / 'assets/results'}")
    if any(missing.values()):
        print("Some featured images are missing:")
        for part, names in missing.items():
            if names:
                print(f"  {part}: {', '.join(names)}")
        print("Check that both notebooks finished with GROUP='all' and that you used their final export ZIPs.")
    else:
        print("All featured images are ready. Open index.html to preview the completed site.")


if __name__ == "__main__":
    main()
