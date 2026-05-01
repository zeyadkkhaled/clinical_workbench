"""
config.py
Ownership: Zeyad
Purpose: Store shared constants, app title, default window size, supported formats, and default AI model paths.
"""

APP_TITLE = "Clinical Image Analysis Workbench"
DEFAULT_WINDOW_SIZE = "1200x800"

SUPPORTED_FORMATS = [
    ("All supported", "*.jpg *.jpeg *.bmp *.png *.dcm"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("PNG", "*.png"),
    ("DICOM", "*.dcm")
]

DEFAULT_AI_MODEL_PATH = "models/segmentation_model.pth"

# TODO (Zeyad): Add any other shared UI or theme constants here
