import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from morphology_engine import apply_threshold, opening, closing

# Load image
img = Image.open("test.jpg").convert('L')  
img = np.array(img)

# Step 1: Threshold
binary = apply_threshold(img, 128)

# FIX: ensure proper types
binary = binary.astype(np.bool_)


# Step 2: Structuring Element
SE = np.ones((3,3))
SE = SE.astype(np.bool_)

print(binary.dtype, SE.dtype)
# Step 3: Opening & Closing
open_img = opening(binary, SE)
close_img = closing(binary, SE)

# Display
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(binary, cmap='gray')
plt.title("Binary")

plt.subplot(2,2,3)
plt.imshow(open_img, cmap='gray')
plt.title("Opening")

plt.subplot(2,2,4)
plt.imshow(close_img, cmap='gray')
plt.title("Closing")

plt.show()