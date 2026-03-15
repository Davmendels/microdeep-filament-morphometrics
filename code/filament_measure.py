"""
All measurements are returned in pixel units.
Physical calibration is intentionally left to the user.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FilamentMeasurement:
    area_px2: float
    perimeter_px: float
    length_px: float
    diameter_px: float
    aspect_ratio: float
    valid: bool
    reason: str = ""


def estimate_length_diameter(area: float, perimeter: float) -> Tuple[float, float]:
    """
    Estimate filament length and diameter from contour area and perimeter.

    Model:
        x^2 - (P/2)x + A = 0

    The larger root is interpreted as filament length, the smaller as diameter.
    """
    if area <= 0:
        raise ValueError("Area must be positive.")
    if perimeter <= 0:
        raise ValueError("Perimeter must be positive.")

    discriminant = perimeter**2 - 16.0 * area
    if discriminant < 0:
        raise ValueError(
            f"Negative discriminant ({discriminant:.6f}); contour outside model assumptions."
        )

    sqrt_disc = math.sqrt(discriminant)
    root1 = 0.25 * (perimeter + sqrt_disc)
    root2 = 0.25 * (perimeter - sqrt_disc)

    return max(root1, root2), min(root1, root2)


def threshold_grayscale_image(
    gray: np.ndarray,
    block_size: int = 51,
    c: int = 10,
    invert: bool = True,
) -> np.ndarray:
    """
    Threshold a grayscale image using Gaussian adaptive thresholding.
    """
    if gray.ndim != 2:
        raise ValueError("Expected a single-channel grayscale image.")

    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size must be an odd integer >= 3.")

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        c,
    )


def morphological_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a morphological closing (dilation followed by erosion).
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")

    if mask.dtype != np.uint8:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def read_image(path: str, as_gray: bool) -> np.ndarray:
    """
    Read an image from disk.
    """
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_UNCHANGED
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def contour_aspect_ratio(contour: np.ndarray) -> float:
    """
    Estimate contour aspect ratio from its bounding rectangle.
    """
    x, y, w, h = cv2.boundingRect(contour)
    short_side = max(1, min(w, h))
    long_side = max(w, h)
    return float(long_side) / float(short_side)


def largest_valid_contour_from_mask(
    mask: np.ndarray,
    min_area: float = 50.0,
    min_aspect_ratio: float = 2.0,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Extract contours from a binary mask, rank them by area, and keep the largest
    contour satisfying minimum area and aspect ratio constraints.
    """
    if mask.dtype != np.uint8:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    else:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No contour found in mask."

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        ar = contour_aspect_ratio(contour)
        if ar < min_aspect_ratio:
            continue

        return contour, ""

    return None, (
        f"No contour passed filtering (min_area={min_area}, "
        f"min_aspect_ratio={min_aspect_ratio})."
    )


def measure_filament_from_contour(contour: np.ndarray) -> FilamentMeasurement:
    """
    Measure a filament from a contour.
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    aspect_ratio = contour_aspect_ratio(contour)

    try:
        length_px, diameter_px = estimate_length_diameter(area, perimeter)
        return FilamentMeasurement(
            area_px2=area,
            perimeter_px=perimeter,
            length_px=length_px,
            diameter_px=diameter_px,
            aspect_ratio=aspect_ratio,
            valid=True,
        )
    except ValueError as exc:
        return FilamentMeasurement(
            area_px2=area,
            perimeter_px=perimeter,
            length_px=float("nan"),
            diameter_px=float("nan"),
            aspect_ratio=aspect_ratio,
            valid=False,
            reason=str(exc),
        )


def measure_filament_from_mask(
    mask: np.ndarray,
    min_area: float = 50.0,
    min_aspect_ratio: float = 2.0,
) -> FilamentMeasurement:
    """
    Measure a filament from a binary mask.
    """
    contour, reason = largest_valid_contour_from_mask(
        mask,
        min_area=min_area,
        min_aspect_ratio=min_aspect_ratio,
    )

    if contour is None:
        return FilamentMeasurement(
            area_px2=0.0,
            perimeter_px=0.0,
            length_px=float("nan"),
            diameter_px=float("nan"),
            aspect_ratio=float("nan"),
            valid=False,
            reason=reason,
        )

    return measure_filament_from_contour(contour)


def print_measurement(measurement: FilamentMeasurement) -> None:
    """
    Print a compact measurement summary.
    """
    if measurement.valid:
        print(
            f"valid={measurement.valid} "
            f"area_px2={measurement.area_px2:.3f} "
            f"perimeter_px={measurement.perimeter_px:.3f} "
            f"length_px={measurement.length_px:.3f} "
            f"diameter_px={measurement.diameter_px:.3f} "
            f"aspect_ratio={measurement.aspect_ratio:.3f}"
        )
    else:
        print(
            f"valid={measurement.valid} "
            f"area_px2={measurement.area_px2:.3f} "
            f"perimeter_px={measurement.perimeter_px:.3f} "
            f"reason=\"{measurement.reason}\""
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure filament length and diameter in pixel units."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--mode",
        choices=["gray", "mask"],
        default="gray",
        help="Interpret input as grayscale image or binary mask",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=51,
        help="Adaptive threshold block size (odd integer, grayscale mode only)",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=10,
        help="Adaptive threshold constant (grayscale mode only)",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Do not invert threshold polarity in grayscale mode",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        help="Morphological closing kernel size",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=50.0,
        help="Minimum contour area in pixels^2",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=2.0,
        help="Minimum contour aspect ratio",
    )
    parser.add_argument(
        "--save-mask",
        default="",
        help="Optional path to save the processed mask",
    )

    args = parser.parse_args()

    if args.mode == "mask":
        mask = read_image(args.image, as_gray=True)
        mask = morphological_close(mask, kernel_size=args.close_kernel)

        measurement = measure_filament_from_mask(
            mask,
            min_area=args.min_area,
            min_aspect_ratio=args.min_aspect_ratio,
        )
        print_measurement(measurement)
        return

    gray = read_image(args.image, as_gray=True)
    mask = threshold_grayscale_image(
        gray,
        block_size=args.block_size,
        c=args.c,
        invert=not args.no_invert,
    )
    mask = morphological_close(mask, kernel_size=args.close_kernel)

    if args.save_mask:
        cv2.imwrite(args.save_mask, mask)

    measurement = measure_filament_from_mask(
        mask,
        min_area=args.min_area,
        min_aspect_ratio=args.min_aspect_ratio,
    )
    print_measurement(measurement)


if __name__ == "__main__":
    main()