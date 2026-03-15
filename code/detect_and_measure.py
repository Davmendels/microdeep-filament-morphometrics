from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from filament_measure import (
    threshold_grayscale_image,
    morphological_close,
    measure_filament_from_mask,
)


def list_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return [
        p for p in sorted(input_path.iterdir())
        if p.suffix.lower() in exts
    ]


def crop_roi(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]


def ensure_dir(path: str | None):
    if path:
        os.makedirs(path, exist_ok=True)


def main():

    parser = argparse.ArgumentParser(
        description="YOLO detection + filament morphometrics"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="results.csv")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

    parser.add_argument("--classes", nargs="+", default=None)

    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--c", type=int, default=10)
    parser.add_argument("--close-kernel", type=int, default=3)

    parser.add_argument("--min-area", type=float, default=50)
    parser.add_argument("--min-aspect-ratio", type=float, default=2)

    parser.add_argument("--save-rois", default=None)
    parser.add_argument("--save-masks", default=None)

    args = parser.parse_args()

    ensure_dir(args.save_rois)
    ensure_dir(args.save_masks)

    model = YOLO(args.weights)

    images = list_images(Path(args.input))

    rows = []

    for image_path in images:

        print(f"Processing {image_path.name}")

        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for det_id, (box, score, cls) in enumerate(zip(boxes, scores, classes)):

                class_name = model.names[cls]

                if args.classes and class_name not in args.classes:
                    continue

                roi = crop_roi(gray, box)

                mask = threshold_grayscale_image(
                    roi,
                    block_size=args.block_size,
                    c=args.c,
                )

                mask = morphological_close(mask, args.close_kernel)

                measurement = measure_filament_from_mask(
                    mask,
                    min_area=args.min_area,
                    min_aspect_ratio=args.min_aspect_ratio,
                )

                if args.save_rois:
                    roi_name = f"{image_path.stem}_{det_id}.png"
                    cv2.imwrite(
                        str(Path(args.save_rois) / roi_name),
                        roi,
                    )

                if args.save_masks:
                    mask_name = f"{image_path.stem}_{det_id}.png"
                    cv2.imwrite(
                        str(Path(args.save_masks) / mask_name),
                        mask,
                    )

                row = {
                    "image": image_path.name,
                    "det_id": det_id,
                    "class": class_name,
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "area_px2": measurement.area_px2,
                    "perimeter_px": measurement.perimeter_px,
                    "length_px": measurement.length_px,
                    "diameter_px": measurement.diameter_px,
                    "aspect_ratio": measurement.aspect_ratio,
                    "valid": measurement.valid,
                    "reason": measurement.reason,
                }

                rows.append(row)

    if rows:

        keys = rows[0].keys()

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Processed {len(images)} images")
    print(f"Wrote {len(rows)} detections to {args.output}")


if __name__ == "__main__":
    main()
