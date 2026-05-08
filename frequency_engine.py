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


# Youssra-Phase2
def _to_grayscale_float(image_array):
    """Convert grayscale/RGB/RGBA image data to grayscale float64 for Fourier matching."""
    if not _is_valid_image_array(image_array):
        raise ValueError("Invalid image array.")

    array = _to_uint8(image_array)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.shape[2] == 1:
        return array[:, :, 0].astype(np.float64)

    red = array[:, :, 0].astype(np.float64)
    green = array[:, :, 1].astype(np.float64)
    blue = array[:, :, 2].astype(np.float64)
    return 0.299 * red + 0.587 * green + 0.114 * blue


# Youssra-Phase2
def _window_sum(image, window_h, window_w):
    """Compute all valid HxW window sums with a summed-area table."""
    padded = np.pad(image, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    return (
        integral[window_h:, window_w:]
        - integral[:-window_h, window_w:]
        - integral[window_h:, :-window_w]
        + integral[:-window_h, :-window_w]
    )


# Youssra-Phase2 START: Fourier template matching
def template_matching_fourier(image_array, template_array):
    """
    Locate a cropped template inside an image using Fourier-domain cross-correlation.

    Returns a dictionary with top-left/bottom-right coordinates, peak score, and
    the valid response map. Coordinates are clamped inside the source image.
    """
    image = _to_grayscale_float(image_array)
    template = _to_grayscale_float(template_array)

    image_h, image_w = image.shape
    template_h, template_w = template.shape

    if template_h <= 0 or template_w <= 0:
        raise ValueError("Template is empty.")
    if template_h > image_h or template_w > image_w:
        raise ValueError("Template must not be larger than the image.")

    template_zero_mean = template - np.mean(template)
    if not np.any(template_zero_mean):
        raise ValueError("Template has no intensity variation.")

    padded_template = np.zeros_like(image, dtype=np.float64)
    padded_template[:template_h, :template_w] = template_zero_mean

    f_image = np.fft.fft2(image)
    f_template = np.fft.fft2(padded_template)
    numerator = np.real(np.fft.ifft2(f_image * np.conj(f_template)))

    valid_h = image_h - template_h + 1
    valid_w = image_w - template_w + 1
    valid_numerator = numerator[:valid_h, :valid_w]

    window_area = float(template_h * template_w)
    window_sum = _window_sum(image, template_h, template_w)
    window_sumsq = _window_sum(image * image, template_h, template_w)
    window_variance_sum = np.maximum(window_sumsq - (window_sum * window_sum / window_area), 0.0)
    template_energy = np.sum(template_zero_mean * template_zero_mean)
    denominator = np.sqrt(window_variance_sum * template_energy)

    valid_response = np.zeros_like(valid_numerator, dtype=np.float64)
    np.divide(
        valid_numerator,
        denominator,
        out=valid_response,
        where=denominator > 1e-12,
    )
    peak_y, peak_x = np.unravel_index(np.argmax(valid_response), valid_response.shape)

    x1 = int(np.clip(peak_x, 0, image_w - 1))
    y1 = int(np.clip(peak_y, 0, image_h - 1))
    x2 = int(np.clip(x1 + template_w, 0, image_w))
    y2 = int(np.clip(y1 + template_h, 0, image_h))

    return {
        "top_left": (x1, y1),
        "bottom_right": (x2, y2),
        "score": float(valid_response[peak_y, peak_x]),
        "response": valid_response,
    }
# Youssra-Phase2 END: Fourier template matching
