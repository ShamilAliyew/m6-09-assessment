from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from detector import CatDetector


APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CSV_FIELDS = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]


def info() -> None:
    with (APP_DIR / "STUDENT.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    print(json.dumps(payload, indent=2))


def predict() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = CatDetector(APP_DIR / "models" / "best.onnx")
    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for image_path in image_paths:
            rel_path = image_path.relative_to(INPUT_DIR).as_posix()
            detections = detector.predict(image_path)
            if not detections:
                writer.writerow({"image_path": rel_path})
                continue

            for detection in detections:
                writer.writerow({"image_path": rel_path, **detection})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat detector assessment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info")
    subparsers.add_parser("predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "info":
        info()
    elif args.command == "predict":
        predict()


if __name__ == "__main__":
    main()
