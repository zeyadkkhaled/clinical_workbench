"""
spatial_engine.py
Purpose: Spatial filtering and geometric transformations from scratch.
"""

import numpy as np


def _is_valid_image_array(image_array):
    """
    Checks whether the input is a non-empty grayscale or color image array.
    """
    try:
        if image_array is None:
            return False

        array = np.asarray(image_array)
        if array.size == 0 or array.ndim not in (2, 3):
            return False

        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            return False

        return array.shape[0] > 0 and array.shape[1] > 0
    except Exception:
        return False


def _to_uint8(image_array):
    """
    Converts an image array to uint8 using safe clipping/min-max normalization.
    """
    try:
        array = np.asarray(image_array)
        if array.size == 0:
            return None

        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

        if array.dtype == np.uint8:
            return array.copy()

        if np.issubdtype(array.dtype, np.integer):
            return np.clip(array, 0, 255).astype(np.uint8)

        array = array.astype(np.float64)
        min_value = np.min(array)
        max_value = np.max(array)

        if max_value == min_value:
            return np.zeros(array.shape, dtype=np.uint8)

        normalized = (array - min_value) * 255.0 / (max_value - min_value)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    except Exception:
        return None


def _rgb_to_grayscale(image_array):
    """
    Converts RGB/RGBA input to grayscale using the standard luminance formula.
    """
    array = _to_uint8(image_array)
    if array is None:
        return None

    if array.ndim == 2:
        return array

    if array.shape[2] == 1:
        return array[:, :, 0]

    red = array[:, :, 0].astype(np.float64)
    green = array[:, :, 1].astype(np.float64)
    blue = array[:, :, 2].astype(np.float64)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.clip(gray, 0, 255).astype(np.uint8)


def _manual_histogram(block):
    """
    Computes a 256-bin histogram manually for a uint8 image block.
    """
    histogram = np.zeros(256, dtype=np.int64)
    flat_block = block.ravel()

    for value in flat_block:
        histogram[int(value)] += 1

    return histogram


def _equalize_block(block):
    """
    Applies histogram equalization to one local uint8 block.
    """
    histogram = _manual_histogram(block)
    cdf = np.zeros(256, dtype=np.int64)

    running_total = 0
    for intensity in range(256):
        running_total += histogram[intensity]
        cdf[intensity] = running_total

    nonzero_cdf = cdf[cdf > 0]
    if nonzero_cdf.size == 0:
        return block.copy()

    cdf_min = nonzero_cdf[0]
    total_pixels = block.size

    if total_pixels == cdf_min:
        return block.copy()

    mapping = np.zeros(256, dtype=np.uint8)
    for intensity in range(256):
        equalized_value = (cdf[intensity] - cdf_min) * 255.0 / (total_pixels - cdf_min)
        mapping[intensity] = np.uint8(np.clip(round(equalized_value), 0, 255))

    return mapping[block]


def zoom_nearest_neighbor(image_array, scale):
    """
    Zooms an image using nearest-neighbor interpolation from scratch.

    Args:
        image_array (array-like): Grayscale HxW or color HxWxC image.
        scale (float): Positive zoom factor. Values above 1 zoom in, values
            between 0 and 1 zoom out.

    Returns:
        numpy.ndarray | None: Zoomed uint8 image, or None for invalid input.
    """
    try:
        if not _is_valid_image_array(image_array) or scale <= 0:
            return None

        image = _to_uint8(image_array)
        if image is None:
            return None

        height, width = image.shape[:2]
        output_height = max(1, int(round(height * scale)))
        output_width = max(1, int(round(width * scale)))

        if image.ndim == 2:
            zoomed = np.zeros((output_height, output_width), dtype=np.uint8)
        else:
            zoomed = np.zeros((output_height, output_width, image.shape[2]), dtype=np.uint8)

        for output_y in range(output_height):
            source_y = output_y / scale
            nearest_y = int(round(source_y))
            nearest_y = min(max(nearest_y, 0), height - 1)

            for output_x in range(output_width):
                source_x = output_x / scale
                nearest_x = int(round(source_x))
                nearest_x = min(max(nearest_x, 0), width - 1)

                zoomed[output_y, output_x] = image[nearest_y, nearest_x]

        return zoomed
    except Exception:
        return None

def zoom_linear(image_array, scale):
    """
    Zooms an image using manual bilinear interpolation.

    The function uses backward mapping so each output pixel samples from a
    floating-point source coordinate in the original image.
    """
    try:
        if not _is_valid_image_array(image_array) or scale <= 0:
            return None

        image = _to_uint8(image_array)
        if image is None:
            return None

        source = image.astype(np.float64)
        height, width = source.shape[:2]
        output_height = max(1, int(round(height * scale)))
        output_width = max(1, int(round(width * scale)))

        if image.ndim == 2:
            zoomed = np.zeros((output_height, output_width), dtype=np.float64)
        else:
            zoomed = np.zeros((output_height, output_width, image.shape[2]), dtype=np.float64)

        for output_y in range(output_height):
            source_y = output_y / scale
            y1 = int(np.floor(source_y))
            y2 = min(y1 + 1, height - 1)
            y1 = min(max(y1, 0), height - 1)
            dy = source_y - y1

            for output_x in range(output_width):
                source_x = output_x / scale
                x1 = int(np.floor(source_x))
                x2 = min(x1 + 1, width - 1)
                x1 = min(max(x1, 0), width - 1)
                dx = source_x - x1

                top_left = source[y1, x1]
                top_right = source[y1, x2]
                bottom_left = source[y2, x1]
                bottom_right = source[y2, x2]

                top = (1.0 - dx) * top_left + dx * top_right
                bottom = (1.0 - dx) * bottom_left + dx * bottom_right
                zoomed[output_y, output_x] = (1.0 - dy) * top + dy * bottom

        return np.clip(np.rint(zoomed), 0, 255).astype(np.uint8)
    except Exception:
        return None

def local_histogram_equalization(image_array, block_size):
    """
    Applies local histogram equalization block by block from scratch.

    RGB/RGBA input is converted to grayscale first using:
        gray = 0.299*R + 0.587*G + 0.114*B
    """
    try:
        if not _is_valid_image_array(image_array) or block_size <= 0:
            return None

        block_size = int(block_size)
        if block_size <= 0:
            return None

        gray = _rgb_to_grayscale(image_array)
        if gray is None:
            return None

        height, width = gray.shape
        enhanced = np.zeros((height, width), dtype=np.uint8)

        for row_start in range(0, height, block_size):
            row_end = min(row_start + block_size, height)

            for col_start in range(0, width, block_size):
                col_end = min(col_start + block_size, width)

                block = gray[row_start:row_end, col_start:col_end]
                enhanced[row_start:row_end, col_start:col_end] = _equalize_block(block)

        return enhanced
    except Exception:
        return None

def apply_2d_convolution(image_array, kernel):
    """Applies a 2D convolution with a given kernel."""
    # TODO (Zeyad): Implement 2D convolution from scratch
    return None

def apply_smoothing_filter(image_array):
    """Applies a spatial smoothing filter."""
    # TODO (Zeyad): Implement smoothing filter using convolution
    return None

def apply_edge_detection(image_array):
    """Applies a spatial edge detection filter."""
    # TODO (Zeyad): Implement edge detection using convolution
    return None

def apply_median_filter(image_array):
    """Applies a median filter."""
    # TODO (Zeyad): Implement median filter
    return None

def rotate_image(image_array, angle):
    """Rotates the image by a given angle."""
    # TODO (Youssra): Implement rotation with custom bilinear interpolation
    return None

def shear_image(image_array, shear_factor):
    """Shears the image by a given factor."""
    # TODO (Youssra): Implement shearing with custom bilinear interpolation
    return None
