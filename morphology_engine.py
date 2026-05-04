"""
morphology_engine.py
Purpose: Morphological operations from scratch.
"""
import numpy as np

def erode(image_array, structuring_element):
    """Performs morphological erosion."""
    # TODO (Zeyad): Implement base spatial array logic for erosion/dilation
    return None

def dilate(image_array, structuring_element):
    """Performs morphological dilation."""
    # TODO (Zeyad): Implement base spatial array logic for erosion/dilation
    return None

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
    

def opening(image_array, structuring_element):
    """Performs morphological opening."""
    # TODO (Youssra): Implement slider binarization and compound logic calling Zeyad's base functions
    # Step 1: Erosion
    eroded = erode(image_array, structuring_element)
    
    # Step 2: Dilation
    opened = dilate(eroded, structuring_element)
    
    return opened
    

def closing(image_array, structuring_element):
    """Performs morphological closing."""
    # TODO (Youssra): Implement slider binarization and compound logic calling Zeyad's base functions
     # Step 1: Dilation
    dilated = dilate(image_array, structuring_element)
    
    # Step 2: Erosion
    closed = erode(dilated, structuring_element)
    
    return closed
    

def extract_boundary(image_array, structuring_element):
    """Extracts the boundary of objects in the image."""
    # TODO (Ahmed): Implement morphological boundary subtraction
    return None
