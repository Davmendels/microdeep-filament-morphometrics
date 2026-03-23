from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class HelixMeasurement:
    skeleton_length_px: float
    coil_count: float
    coil_pitch_px: float
    coil_amplitude_px: float
    length_unwrapped_px: float
    valid: bool
    reason: str = ""


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def clean_mask_from_largest_contour(mask: np.ndarray) -> np.ndarray:
    contour = largest_contour(mask)
    clean = np.zeros_like(mask, dtype=np.uint8)

    if contour is None:
        return clean

    cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)
    return clean


def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


def skeleton_points(skel: np.ndarray) -> np.ndarray:
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((xs, ys)).astype(float)


def pca_rotate(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points.copy()

    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotated = centered @ eigvecs
    return rotated


def estimate_skeleton_length(skel: np.ndarray) -> float:
    """
    First-order skeleton length estimate: count non-zero skeleton pixels.
    """
    return float(np.count_nonzero(skel > 0))


def estimate_coil_parameters(skel: np.ndarray, points: np.ndarray) -> HelixMeasurement:
    if len(points) < 10:
        return HelixMeasurement(
            skeleton_length_px=float("nan"),
            coil_count=float("nan"),
            coil_pitch_px=float("nan"),
            coil_amplitude_px=float("nan"),
            length_unwrapped_px=float("nan"),
            valid=False,
            reason="Not enough skeleton points.",
        )

    pts = pca_rotate(points)
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    x = pts[:, 0]
    y = pts[:, 1]

    skeleton_length = estimate_skeleton_length(skel)

    amplitude = 0.5 * (np.percentile(y, 95) - np.percentile(y, 5))

    if len(y) >= 7:
        kernel = np.ones(7) / 7.0
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y.copy()

    dy = np.diff(y_smooth)
    s = np.sign(dy)
    ds = np.diff(s)

    peak_idx = np.where(ds < 0)[0] + 1
    trough_idx = np.where(ds > 0)[0] + 1
    extrema = np.sort(np.concatenate([peak_idx, trough_idx]))

    if len(extrema) < 3:
        return HelixMeasurement(
            skeleton_length_px=skeleton_length,
            coil_count=float("nan"),
            coil_pitch_px=float("nan"),
            coil_amplitude_px=amplitude,
            length_unwrapped_px=float("nan"),
            valid=False,
            reason="Could not detect enough extrema.",
        )

    dx = np.diff(x[extrema])
    half_period = np.median(dx)
    pitch = 2.0 * half_period

    x_span = x.max() - x.min()
    coil_count = x_span / pitch if pitch > 0 else float("nan")

    if pitch > 0:
        k = 2.0 * math.pi / pitch
        xx = np.linspace(0, x_span, 1000)
        dydx = amplitude * k * np.cos(k * xx)
        unwrapped_length = float(np.trapezoid(np.sqrt(1.0 + dydx ** 2), xx))
    else:
        unwrapped_length = float("nan")

    return HelixMeasurement(
        skeleton_length_px=skeleton_length,
        coil_count=float(coil_count),
        coil_pitch_px=float(pitch),
        coil_amplitude_px=float(amplitude),
        length_unwrapped_px=float(unwrapped_length),
        valid=True,
    )


def print_measurement(measurement: HelixMeasurement) -> None:
    if measurement.valid:
        print(
            f"valid={measurement.valid} "
            f"skeleton_length_px={measurement.skeleton_length_px:.3f} "
            f"coil_count={measurement.coil_count:.3f} "
            f"coil_pitch_px={measurement.coil_pitch_px:.3f} "
            f"coil_amplitude_px={measurement.coil_amplitude_px:.3f} "
            f"length_unwrapped_px={measurement.length_unwrapped_px:.3f}"
        )
    else:
        print(
            f"valid={measurement.valid} "
            f"skeleton_length_px={measurement.skeleton_length_px:.3f} "
            f"coil_amplitude_px={measurement.coil_amplitude_px:.3f} "
            f"reason=\"{measurement.reason}\""
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental skeleton and coil analysis on a binary mask."
    )
    parser.add_argument("mask", help="Path to binary mask image")
    parser.add_argument(
        "--save-clean-mask",
        default="",
        help="Optional path to save cleaned largest-contour mask",
    )
    parser.add_argument(
        "--save-skeleton",
        default="",
        help="Optional path to save skeleton image",
    )
    args = parser.parse_args()

    raw_mask = read_mask(args.mask)
    clean_mask = clean_mask_from_largest_contour(raw_mask)
    skeleton = skeleton_from_mask(clean_mask)
    points = skeleton_points(skeleton)
    measurement = estimate_coil_parameters(skeleton, points)

    print_measurement(measurement)

    if args.save_clean_mask:
        cv2.imwrite(str(Path(args.save_clean_mask)), clean_mask)

    if args.save_skeleton:
        cv2.imwrite(str(Path(args.save_skeleton)), skeleton)


if __name__ == "__main__":
    main()