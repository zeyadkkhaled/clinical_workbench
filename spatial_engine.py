"""
# Shared Module: Spatial Operations & Interpolation
Purpose: Spatial filtering and geometric transformations from scratch.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    Computes a 256-bin histogram manually for a uint8 image block using fast NumPy bincount.
    """
    return np.bincount(block.ravel(), minlength=256).astype(np.int64)


def _equalize_block(block):
    """
    Applies histogram equalization to one local uint8 block using vectorized math.
    """
    histogram = _manual_histogram(block)
    cdf = np.cumsum(histogram).astype(np.int64)

    nonzero_cdf = cdf[cdf > 0]
    if nonzero_cdf.size == 0:
        return block.copy()

    cdf_min = nonzero_cdf[0]
    total_pixels = block.size

    if total_pixels == cdf_min:
        return block.copy()

    equalized_values = (cdf - cdf_min) * 255.0 / (total_pixels - cdf_min)
    mapping = np.clip(np.round(equalized_values), 0, 255).astype(np.uint8)

    return mapping[block]


# Bahr-Phase1-Fix START: Smooth local histogram equalization
def _equalization_mapping(block):
    """Build a 256-value equalization lookup table for one local tile."""
    histogram = _manual_histogram(block)
    cdf = np.cumsum(histogram).astype(np.int64)

    nonzero_cdf = cdf[cdf > 0]
    if nonzero_cdf.size == 0:
        return np.arange(256, dtype=np.uint8)

    cdf_min = nonzero_cdf[0]
    total_pixels = block.size
    if total_pixels == cdf_min:
        return np.arange(256, dtype=np.uint8)

    equalized_values = (cdf - cdf_min) * 255.0 / (total_pixels - cdf_min)
    return np.clip(np.round(equalized_values), 0, 255).astype(np.uint8)
# Bahr-Phase1-Fix END: Smooth local histogram equalization


# Author: Ahmed Hassan Bahr
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

        output_y = np.arange(output_height)
        output_x = np.arange(output_width)

        source_y = output_y / scale
        source_x = output_x / scale

        nearest_y = np.clip(np.round(source_y).astype(int), 0, height - 1)
        nearest_x = np.clip(np.round(source_x).astype(int), 0, width - 1)

        # Advanced indexing with broadcasting saves massive memory compared to meshgrids
        zoomed = image[nearest_y[:, np.newaxis], nearest_x[np.newaxis, :]]

        # Ensure the array is strictly C-contiguous before Pillow processes it
        return np.ascontiguousarray(zoomed)
    except Exception:
        return None

# Author: Ahmed Hassan Bahr
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

        # Use float32 to drastically reduce memory usage and prevent OOM crashes
        source = image.astype(np.float32)
        height, width = source.shape[:2]
        output_height = max(1, int(round(height * scale)))
        output_width = max(1, int(round(width * scale)))

        output_y = np.arange(output_height)
        output_x = np.arange(output_width)

        source_y = output_y / scale
        source_x = output_x / scale

        y1 = np.clip(np.floor(source_y).astype(int), 0, height - 1)
        y2 = np.clip(y1 + 1, 0, height - 1)
        dy = (source_y - y1).astype(np.float32)

        x1 = np.clip(np.floor(source_x).astype(int), 0, width - 1)
        x2 = np.clip(x1 + 1, 0, width - 1)
        dx = (source_x - x1).astype(np.float32)

        y1_idx = y1[:, np.newaxis]
        y2_idx = y2[:, np.newaxis]
        x1_idx = x1[np.newaxis, :]
        x2_idx = x2[np.newaxis, :]

        dy_grid = dy[:, np.newaxis]
        dx_grid = dx[np.newaxis, :]

        if source.ndim == 3:
            dy_grid = dy_grid[:, :, np.newaxis]
            dx_grid = dx_grid[:, :, np.newaxis]

        # Calculate sequentially to save memory! 
        # Creating all 4 full grids simultaneously causes MemoryError on large images.
        zoomed = source[y1_idx, x1_idx] * (1.0 - dx_grid)
        zoomed += source[y1_idx, x2_idx] * dx_grid
        zoomed *= (1.0 - dy_grid)

        bottom = source[y2_idx, x1_idx] * (1.0 - dx_grid)
        bottom += source[y2_idx, x2_idx] * dx_grid
        bottom *= dy_grid

        zoomed += bottom

        return np.ascontiguousarray(np.clip(np.rint(zoomed), 0, 255).astype(np.uint8))
    except Exception:
        return None

# Author: Ahmed Hassan Bahr
def local_histogram_equalization(image_array, block_size):
    """
    Applies smooth local histogram equalization from scratch.

    RGB/RGBA input is converted to grayscale first using:
        gray = 0.299*R + 0.587*G + 0.114*B

    # Bahr-Phase1-Fix START: Smooth local histogram equalization
    This version reduces block artifacts by calculating one histogram/CDF
    mapping per tile, then bilinearly interpolating the four neighboring tile
    mappings for each pixel instead of pasting independently equalized blocks.
    # Bahr-Phase1-Fix END: Smooth local histogram equalization
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
        if block_size <= 1:
            return gray.copy()

        if block_size >= max(height, width):
            return _equalize_block(gray)

        # Bahr-Phase1-Fix START: Smooth local histogram equalization
        row_starts = list(range(0, height, block_size))
        col_starts = list(range(0, width, block_size))
        row_bounds = [(start, min(start + block_size, height)) for start in row_starts]
        col_bounds = [(start, min(start + block_size, width)) for start in col_starts]
        row_centers = np.array([(start + end - 1) / 2.0 for start, end in row_bounds], dtype=np.float64)
        col_centers = np.array([(start + end - 1) / 2.0 for start, end in col_bounds], dtype=np.float64)

        mappings = np.zeros((len(row_bounds), len(col_bounds), 256), dtype=np.uint8)
        for tile_y, (row_start, row_end) in enumerate(row_bounds):
            for tile_x, (col_start, col_end) in enumerate(col_bounds):
                mappings[tile_y, tile_x] = _equalization_mapping(
                    gray[row_start:row_end, col_start:col_end]
                )

        col_positions = np.arange(width, dtype=np.float64)
        x_upper = np.searchsorted(col_centers, col_positions, side="right")
        x_lower = np.clip(x_upper - 1, 0, len(col_centers) - 1)
        x_upper = np.clip(x_upper, 0, len(col_centers) - 1)
        x_denom = col_centers[x_upper] - col_centers[x_lower]
        x_weight = np.divide(
            col_positions - col_centers[x_lower],
            x_denom,
            out=np.zeros(width, dtype=np.float64),
            where=x_denom != 0,
        )
        x_weight = np.clip(x_weight, 0.0, 1.0)

        enhanced = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            y_pos = float(row)
            y_upper = int(np.searchsorted(row_centers, y_pos, side="right"))
            y_lower = max(0, min(y_upper - 1, len(row_centers) - 1))
            y_upper = max(0, min(y_upper, len(row_centers) - 1))
            y_denom = row_centers[y_upper] - row_centers[y_lower]
            y_weight = 0.0 if y_denom == 0 else (y_pos - row_centers[y_lower]) / y_denom
            y_weight = min(1.0, max(0.0, y_weight))

            intensities = gray[row]
            top_left = mappings[y_lower, x_lower, intensities].astype(np.float64)
            top_right = mappings[y_lower, x_upper, intensities].astype(np.float64)
            bottom_left = mappings[y_upper, x_lower, intensities].astype(np.float64)
            bottom_right = mappings[y_upper, x_upper, intensities].astype(np.float64)

            top = top_left * (1.0 - x_weight) + top_right * x_weight
            bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight
            blended = top * (1.0 - y_weight) + bottom * y_weight
            enhanced[row] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

        # Bahr-Phase1-Fix END: Smooth local histogram equalization
        return enhanced
    except Exception:
        return None

# Author: Zeyad Khaled
def apply_2d_convolution(image_array, kernel):
    """
    Applies a 2D convolution with a given kernel using sliding windows.
    
    Clinical Relevance:
        Convolution is the foundational operation for spatial filtering (blurring, sharpening) 
        and feature extraction (edges) in medical images.
        
    Implementation Details:
        - Padding: We use 'edge' padding rather than zero padding. Zero padding introduces 
          artificial black borders which create false edges (ringing artifacts) during convolution. 
          'edge' padding replicates the outermost pixels, maintaining continuity and preventing 
          edge-effect distortions in clinical scans.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image_array, dtype=np.float64)
    
    for i in range(kernel_height):
        for j in range(kernel_width):
            output += padded_image[i:i+image_array.shape[0], j:j+image_array.shape[1]] * kernel[i, j]
            
    return output

# Author: Zeyad Khaled
def apply_smoothing_filter(image_array, filter_type, kernel_size, variance=None):
    """
    Applies a spatial smoothing filter (Average or Gaussian) to reduce noise.
    
    Clinical Relevance:
        Used to suppress quantum mottle (noise) in X-rays or CT scans.
        
    Implementation Details:
        - Average Filter: Uniformly blurs the image. While fast, it severely degrades 
          sharp anatomical edges.
        - Gaussian Filter: Uses a bell-curve (Gaussian) distribution controlled by `variance`.
          A higher variance spreads the weights further out, increasing the blur. Unlike the 
          average filter, it preserves lower-frequency edge structures better while still 
          reducing high-frequency noise.
    """
    if filter_type.lower() == 'average':
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    elif filter_type.lower() == 'gaussian':
        if variance is None:
            variance = 1.0
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        gauss = np.exp(-0.5 * np.square(ax) / variance)
        kernel = np.outer(gauss, gauss)
        kernel /= np.sum(kernel)
    else:
        raise ValueError("Unsupported filter type. Use 'average' or 'gaussian'.")
        
    filtered = apply_2d_convolution(image_array, kernel)
    return np.clip(filtered, 0, 255).astype(image_array.dtype)

# Author: Zeyad Khaled
def apply_edge_detection(image_array, operator_type):
    """
    Applies a spatial edge detection filter (Sobel or Prewitt) using orthogonal gradients.
    
    Clinical Relevance:
        Crucial for highlighting boundaries between different tissues (e.g., bone vs. soft tissue) 
        or detecting the margins of lesions and tumors.
        
    Implementation Details:
        - Matrices: We apply 3x3 gradient operators (kx for horizontal, ky for vertical edges).
          Sobel places a higher weight (2) on the center pixel, offering better noise robustness 
          than Prewitt.
        - Magnitude Calculation: The final edge strength is computed as sqrt(grad_x^2 + grad_y^2). 
          This vector magnitude captures edges in all directions (including diagonals), which 
          is critical since anatomical structures are rarely perfectly horizontal or vertical.
    """
    if operator_type.lower() == 'sobel':
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    elif operator_type.lower() == 'prewitt':
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    else:
        raise ValueError("Unsupported operator. Use 'sobel' or 'prewitt'.")
        
    grad_x = apply_2d_convolution(image_array, kx)
    grad_y = apply_2d_convolution(image_array, ky)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
        
    return np.clip(magnitude, 0, 255).astype(np.uint8)

# Author: Zeyad Khaled
def apply_median_filter(image_array, kernel_size):
    """
    Applies a non-linear median filter.
    
    Clinical Relevance:
        Highly effective at removing 'salt and pepper' noise (isolated extreme pixel values) 
        often seen in corrupted scans or faulty sensor acquisitions, without blurring sharp edges.
        
    Implementation Details:
        - Optimizations: Instead of slow Python nested loops, we use `sliding_window_view` from 
          NumPy to create a 4D view of local neighborhoods, allowing us to compute the median 
          across the entire image simultaneously (axis=(2,3)).
    """
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    windows = sliding_window_view(padded_image, (kernel_size, kernel_size))
    output = np.median(windows, axis=(2, 3))
    return output.astype(image_array.dtype)

# Author: Youssra Hatem
def rotate_image(image_array, angle):
    """Rotates the image by a given angle."""
    # TODO (Youssra): Implement rotation with custom bilinear interpolation
    return None

# Author: Youssra Hatem
def shear_image(image_array, shear_factor):
    """Shears the image by a given factor."""
    # TODO (Youssra): Implement shearing with custom bilinear interpolation
    return None
