# Clinical Image Analysis Workbench - Phase 1 Minimal Test Dataset

This synthetic dataset is designed to test the implemented features described in the project brief.

## Files

### 01_clean_grayscale_detail.png
512x512 grayscale image with gradients, sharp edges, circles, rectangles, fine checker detail, and thin lines.

Tests:
- JPEG/PNG-style image loading
- Zoom: nearest-neighbor and bilinear at 0.5x and 2x
- Average filter
- Gaussian filter, variance 1.0 and 3.0
- Sobel edge detection
- Prewitt edge detection
- Local Histogram Equalization

Expected:
- Zoom should clearly show interpolation differences.
- Average/Gaussian filters should smooth detail.
- Sobel/Prewitt should highlight square, circle, line, and checker edges.
- LHE should enhance local contrast.

### 02_salt_pepper_10_percent.png
Same clean image with 10% salt-and-pepper impulse noise.

Tests:
- Median filter with 3x3 and 5x5 kernels.

Expected:
- Median filter should remove many black/white noise pixels.
- 5x5 should clean more aggressively but blur more detail, because apparently denoising always demands tribute.

### 03_low_contrast_uneven_illumination.png
512x512 low-contrast image with uneven illumination and subtle structures.

Tests:
- Local Histogram Equalization, block sizes 8, 16, 32.

Expected:
- Local contrast should increase.
- Smaller blocks may show stronger local enhancement or block artifacts.
- Larger blocks should look smoother but less locally aggressive.

### 04_binary_morphology_shapes.png
256x256 binary image with white blobs on black background, small holes, dots, and a thin bridge.

Tests:
- Threshold at 127
- Erosion and dilation
- Square and Cross structuring elements, 3x3 and 5x5.

Expected:
- Erosion shrinks objects, removes tiny dots, widens holes, and may break the thin bridge.
- Dilation expands objects, fills small gaps/holes, and thickens structures.
- Square kernels affect diagonals/corners more than Cross kernels.

### 05_clean_grayscale_detail.bmp
BMP version of the clean grayscale image.

Tests:
- BMP loading path.

Expected:
- Should load and behave similarly to the PNG.

### 06_unsupported_format.txt
Plain text file pretending to be relevant. Humanity was a mistake.

Tests:
- Unsupported format error handling.

Expected:
- App should reject it cleanly without crashing.

### 07_corrupted_image.png
Invalid PNG bytes with a PNG extension.

Tests:
- Corrupted image error handling.

Expected:
- App should reject it cleanly without crashing.

### 08_DICOM_generation_instructions.txt
Instructions for generating a valid DICOM file using pydicom sample data.

Tests:
- DICOM loading
- Metadata extraction

Expected:
- With pydicom CT_small.dcm, app should load a CT image and extract available metadata.

## Suggested Test Parameters

- Zoom:
  - Scale 0.5
  - Scale 2.0
  - Nearest-neighbor
  - Bilinear
  - Invalid scale 0 for error handling
  - Very high scale above cap to verify scale cap behavior

- Filters:
  - Average: 3x3, 5x5
  - Gaussian: 3x3, 5x5; variance 1.0 and 3.0
  - Median: 3x3, 5x5 on salt-pepper image
  - Sobel/Prewitt: 3x3, 5x5 on clean detail image

- Local Histogram Equalization:
  - Block sizes 8, 16, 32 on the low-contrast uneven image

- Morphology:
  - Erosion/Dilation
  - Square/Cross
  - Kernel sizes 3x3 and 5x5
