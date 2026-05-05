"""
# Author: Ahmed Hassan Bahr (Data I/O, Interpolation & Statistics) - Complete Ownership
image_io.py
Purpose: Functions to handle loading/saving images, including DICOM metadata.
"""

import os

import numpy as np
from PIL import Image


STANDARD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
SAVE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# Author: Ahmed Hassan Bahr
def normalize_to_uint8(image_array):
    """
    Converts an image array to uint8 safely for display or saving.

    Constant images are mapped to zeros to avoid division by zero. Floating
    point and high bit-depth medical images are min-max normalized into the
    display range 0..255.
    """
    try:
        array = np.asarray(image_array)

        if array.size == 0:
            return None

        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

        if array.dtype == np.uint8:
            return array.copy()

        array = array.astype(np.float64)
        min_value = np.min(array)
        max_value = np.max(array)

        if max_value == min_value:
            return np.zeros(array.shape, dtype=np.uint8)

        normalized = (array - min_value) * 255.0 / (max_value - min_value)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    except Exception:
        return None


def prepare_image_for_display(image_array):
    """
    Prepares grayscale, RGB, or RGBA image data as uint8 for GUI display.
    """
    return normalize_to_uint8(image_array)


# Author: Ahmed Hassan Bahr
def extract_dicom_metadata(ds):
    """
    Extracts useful DICOM metadata while tolerating missing tags.
    """
    def get_value(tag_name):
        value = getattr(ds, tag_name, "N/A")
        if value is None or value == "":
            return "N/A"
        return str(value)

    rows = getattr(ds, "Rows", "N/A")
    columns = getattr(ds, "Columns", "N/A")
    bits = getattr(ds, "BitsAllocated", getattr(ds, "BitsStored", "N/A"))

    return {
        "Patient Name": get_value("PatientName"),
        "Patient Age": get_value("PatientAge"),
        "Modality": get_value("Modality"),
        "Body Part Examined": get_value("BodyPartExamined"),
        "Width": columns if columns is not None else "N/A",
        "Height": rows if rows is not None else "N/A",
        "Bit Depth": bits if bits is not None else "N/A",
        "Samples Per Pixel": getattr(ds, "SamplesPerPixel", "N/A"),
        "Photometric Interpretation": get_value("PhotometricInterpretation"),
    }


# Author: Ahmed Hassan Bahr
def get_standard_image_metadata(image, image_array, file_path):
    """
    Builds metadata for standard image formats loaded by Pillow.
    """
    height, width = image_array.shape[:2]
    channels = 1 if image_array.ndim == 2 else image_array.shape[2]

    return {
        "File Name": os.path.basename(file_path),
        "File Type": os.path.splitext(file_path)[1].upper().replace(".", ""),
        "Width": width,
        "Height": height,
        "Channels": channels,
        "Bit Depth": image_array.dtype.itemsize * 8,
        "Image Mode": image.mode,
    }


# Author: Ahmed Hassan Bahr
def load_image(file_path):
    """
    Loads a standard image or DICOM file.

    Returns:
        tuple: (image_array, metadata_dict, error_message)

    On success, error_message is None. On failure, image_array is None,
    metadata_dict is empty, and error_message is a friendly explanation.
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return None, {}, "Please provide a valid image file path."

        if not os.path.exists(file_path):
            return None, {}, "The selected file does not exist."

        extension = os.path.splitext(file_path)[1].lower()

        if extension in STANDARD_IMAGE_EXTENSIONS:
            try:
                with Image.open(file_path) as image:
                    image.load()

                    # Force convert to grayscale to ensure pure NumPy engines only receive 2D arrays
                    if image.mode != "L":
                        image = image.convert("L")

                    image_array = np.asarray(image)
                    metadata = get_standard_image_metadata(
                        image, image_array, file_path
                    )
                    return image_array, metadata, None
            except Exception:
                return None, {}, "Could not load this image. The file may be corrupted or unsupported."

        if extension in DICOM_EXTENSIONS:
            try:
                import pydicom
            except ImportError:
                return None, {}, "DICOM loading requires pydicom, but it is not installed."

            try:
                ds = pydicom.dcmread(file_path, force=True)

                if not hasattr(ds, "pixel_array"):
                    return None, {}, "This DICOM file does not contain readable pixel data."

                pixel_array = np.asarray(ds.pixel_array)

                if pixel_array.size == 0:
                    return None, {}, "This DICOM file contains empty pixel data."

                display_array = normalize_to_uint8(pixel_array)
                if display_array is None:
                    return None, {}, "Could not convert DICOM pixel data for display."

                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    display_array = 255 - display_array

                metadata = extract_dicom_metadata(ds)
                return display_array, metadata, None
            except Exception:
                return None, {}, "Could not load this DICOM file. It may be corrupted or missing required data."

        return None, {}, "Unsupported file format. Please choose JPEG, JPG, PNG, BMP, or DICOM."
    except Exception:
        return None, {}, "An unexpected error occurred while loading the image."

# Author: Ahmed Hassan Bahr
def save_image(image_array, file_path):
    """
    Saves a NumPy image array as PNG, JPEG/JPG, or BMP.

    Returns:
        tuple: (success_boolean, message)
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return False, "Please provide a valid output file path."

        extension = os.path.splitext(file_path)[1].lower()
        if extension not in SAVE_EXTENSIONS:
            return False, "Please save the image as PNG, JPEG, JPG, or BMP."

        if image_array is None:
            return False, "There is no image to save."

        array = np.asarray(image_array)
        if array.size == 0 or array.ndim not in (2, 3):
            return False, "The image data is invalid and cannot be saved."

        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            return False, "Only grayscale, RGB, or RGBA images can be saved."

        prepared = normalize_to_uint8(array)
        if prepared is None:
            return False, "The image data could not be converted for saving."

        if prepared.ndim == 3 and prepared.shape[2] == 1:
            prepared = prepared[:, :, 0]

        image = Image.fromarray(prepared)

        # JPEG and BMP cannot store alpha in the usual GUI workflow.
        if extension in {".jpg", ".jpeg", ".bmp"} and image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(file_path)
        return True, "Image saved successfully."
    except Exception:
        return False, "Could not save the image. Please check the path and file format."
