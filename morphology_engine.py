"""
# Shared Module: Morphological Operations
Purpose: Morphological operations from scratch.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Author: Zeyad Khaled
def generate_structuring_element(shape_type, size):
    """
    Helper to generate Square and Cross Structuring Elements.
    """
    if size % 2 == 0:
        size += 1
        
    if shape_type.lower() == 'square':
        return np.ones((size, size), dtype=bool)
    elif shape_type.lower() == 'cross':
        se = np.zeros((size, size), dtype=bool)
        center = size // 2
        se[center, :] = True
        se[:, center] = True
        return se
    else:
        raise ValueError("Unsupported shape type. Use 'square' or 'cross'.")

# Author: Zeyad Khaled
def erode(image_array, structuring_element):
    """
    Performs morphological erosion using a specified structuring element (SE).
    
    Clinical Relevance:
        Erosion strips away the outermost layer of pixels from binary objects. It is used to 
        clean up 'salt' noise (small white artifacts) from thresholded masks, separate touching 
        tumors, or shrink segmented regions to ensure we are strictly within tissue boundaries.
        
    Implementation Details:
        - The SE defines the neighborhood. If all pixels under the SE are foreground, the 
          center pixel is kept; otherwise it becomes background.
    """
    is_uint8 = image_array.dtype == np.uint8
    bool_array = image_array > 127 if is_uint8 else image_array.astype(bool)
        
    k_h, k_w = structuring_element.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    padded = np.pad(bool_array, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=True)
    windows = sliding_window_view(padded, (k_h, k_w))
    
    se_sum = structuring_element.sum()
    match = np.sum(windows & structuring_element, axis=(2, 3))
    output_bool = (match == se_sum)
    
    return (output_bool * 255).astype(np.uint8) if is_uint8 else output_bool

# Author: Zeyad Khaled
def dilate(image_array, structuring_element):
    """
    Performs morphological dilation using a specified structuring element (SE).
    
    Clinical Relevance:
        Dilation adds a layer of pixels to the boundaries of binary objects. It is used to 
        clean up 'pepper' noise (small black holes inside foreground objects), bridge gaps in 
        fractured bone scans, or expand segmented regions.
        
    Implementation Details:
        - The SE defines the neighborhood. If any pixel under the SE is foreground, the 
          center pixel becomes foreground.
    """
    is_uint8 = image_array.dtype == np.uint8
    bool_array = image_array > 127 if is_uint8 else image_array.astype(bool)
        
    k_h, k_w = structuring_element.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    padded = np.pad(bool_array, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=False)
    windows = sliding_window_view(padded, (k_h, k_w))
    
    match = np.any(windows & structuring_element, axis=(2, 3))
    
    return (match * 255).astype(np.uint8) if is_uint8 else match

# Author: Youssra Hatem
def apply_threshold(image_array, threshold_value):
    """Binarizes the image based on a threshold."""
    # TODO (Youssra): Implement slider binarization and compound logic calling Zeyad's base functions
    # Ensure input is numpy array
    image_array = np.array(image_array)

    # Create output array
    binary_image = np.zeros_like(image_array, dtype=np.uint8)

    # Apply threshold
    rows, cols = image_array.shape
    for i in range(rows):
        for j in range(cols):
            if image_array[i, j] > threshold_value:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image

# Author: Youssra Hatem
def opening(image_array, structuring_element):
    """Performs morphological opening."""
    # TODO (Youssra): Implement slider binarization and compound logic calling Zeyad's base functions

    # Step 1: Erosion
    eroded = erode(image_array, structuring_element)

    # Step 2: Dilation
    opened = dilate(eroded, structuring_element)

    return opened

# Author: Youssra Hatem
def closing(image_array, structuring_element):
    """Performs morphological closing."""
    # TODO (Youssra): Implement slider binarization and compound logic calling Zeyad's base functions

    # Step 1: Dilation
    dilated = dilate(image_array, structuring_element)

    # Step 2: Erosion
    closed = erode(dilated, structuring_element)

    return closed

def extract_boundary(image_array, structuring_element):
    """
    Extracts the boundary of objects in the image.
    
    Clinical Relevance:
        Isolates the outline of segmented anatomical structures (like a lung or a tumor). 
        This boundary is critical for calculating morphological features such as perimeter, 
        smoothness, or irregularity (which often indicates malignancy).
        
    Implementation Details:
        - The boundary is obtained by subtracting the eroded image from the original image.
          (Note: Base logic is implemented via UI wrappers).
    """
    # Owner: Zeyad - Morphology Core
    eroded = erode(image_array, structuring_element)

    if image_array.dtype == np.uint8 and eroded.dtype == np.uint8:
        return np.clip(
            image_array.astype(np.int16) - eroded.astype(np.int16), 0, 255
        ).astype(np.uint8)

    return (image_array.astype(bool) & ~eroded.astype(bool)).astype(np.uint8) * 255
