"""
# Shared Module: Frequency Domain & ROI (Phase 2)
Purpose: Frequency domain operations, noise modeling, and ROI statistics.
"""

import numpy as np

def generate_notch_filter(shape, center, radius):
    """Generates a notch filter mask."""
    # TODO (Zeyad): Implement Notch filter generation and inverse FFT
    return None

def apply_notch_filter(image_array, filter_mask):
    """Applies a notch filter to the image in the frequency domain."""
    # TODO (Zeyad): Implement Notch filter application and inverse FFT
    return None

# Bahr-Phase2
def _is_valid_image_array(image_array):
    """Return True for non-empty grayscale or RGB/RGBA-style arrays."""
    try:
        array = np.asarray(image_array)
        if array.size == 0 or array.ndim not in (2, 3):
            return False
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            return False
        return array.shape[0] > 0 and array.shape[1] > 0
    except Exception:
        return False


# Bahr-Phase2
def _to_uint8(image_array):
    """Convert image-like data to uint8 by clipping into the display range."""
    array = np.asarray(image_array)
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(array, 0, 255).astype(np.uint8)


# Bahr-Phase2
def inject_noise(image_array, noise_type, parameters=None):
    """
    Inject Gaussian or Uniform additive noise using NumPy only.

    Args:
        image_array: HxW grayscale or HxWxC color image.
        noise_type: "Gaussian" or "Uniform" (case-insensitive).
        parameters: dict with mean/std for Gaussian, low/high for Uniform.

    Returns:
        numpy.ndarray: uint8 noisy image.

    Raises:
        ValueError: for invalid images, noise types, or parameter values.
    """
    if not _is_valid_image_array(image_array):
        raise ValueError("Invalid image array.")

    parameters = parameters or {}
    kind = str(noise_type).strip().lower()
    image = _to_uint8(image_array).astype(np.float64)

    try:
        if kind == "gaussian":
            mean = float(parameters.get("mean", 0.0))
            std = float(parameters.get("std", 10.0))
            if std < 0:
                raise ValueError("Gaussian std must be non-negative.")
            noise = np.random.normal(mean, std, image.shape)
        elif kind == "uniform":
            low = float(parameters.get("low", -10.0))
            high = float(parameters.get("high", 10.0))
            if high < low:
                raise ValueError("Uniform high must be greater than or equal to low.")
            noise = np.random.uniform(low, high, image.shape)
        else:
            raise ValueError("Unsupported noise type. Use Gaussian or Uniform.")
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Bahr-Phase2
def calculate_roi_statistics(image_array, roi):
    """
    Calculate grayscale statistics for a rectangular ROI.

    ROI format is (x1, y1, x2, y2). Coordinates are clamped inside the image.
    RGB/RGBA input is converted with gray = 0.299R + 0.587G + 0.114B.
    """
    if not _is_valid_image_array(image_array):
        raise ValueError("Invalid image array.")

    try:
        if roi is None or len(roi) != 4:
            raise ValueError("Please select a valid ROI.")
        x1, y1, x2, y2 = [int(round(float(value))) for value in roi]
    except (TypeError, ValueError) as exc:
        raise ValueError("Please select a valid ROI.") from exc

    array = _to_uint8(image_array)
    height, width = array.shape[:2]

    left = max(0, min(x1, x2))
    right = min(width, max(x1, x2))
    top = max(0, min(y1, y2))
    bottom = min(height, max(y1, y2))

    if right <= left or bottom <= top:
        raise ValueError("Please select a valid ROI.")

    roi_pixels = array[top:bottom, left:right]
    if roi_pixels.size == 0:
        raise ValueError("Please select a valid ROI.")

    if roi_pixels.ndim == 3:
        if roi_pixels.shape[2] == 1:
            gray = roi_pixels[:, :, 0]
        else:
            red = roi_pixels[:, :, 0].astype(np.float64)
            green = roi_pixels[:, :, 1].astype(np.float64)
            blue = roi_pixels[:, :, 2].astype(np.float64)
            gray = np.clip(0.299 * red + 0.587 * green + 0.114 * blue, 0, 255).astype(np.uint8)
    else:
        gray = roi_pixels

    flat = gray.ravel()
    histogram = np.bincount(flat, minlength=256).astype(np.int64)

    return {
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "std": float(np.std(flat)),
        "min": int(np.min(flat)),
        "max": int(np.max(flat)),
        "pixel_count": int(flat.size),
        "histogram": histogram,
    }

def template_matching_fourier(image_array, template_array):
    """Performs template matching in the Fourier Domain."""
    # TODO (Youssra): Implement cross-correlation in the Fourier Domain
    return None
