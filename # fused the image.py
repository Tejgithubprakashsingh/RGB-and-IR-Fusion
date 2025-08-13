# Program
from google.colab import files
uploaded = files.upload()

!pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load RGB image (color)
rgb_png = cv2.imread('f6.png')
rgb_png = cv2.cvtColor(rgb_png, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Load IR image (grayscale)
ir_png = cv2.imread('f7.png', cv2.IMREAD_GRAYSCALE)

# Resize IR to match RGB
ir_png = cv2.resize(ir_png, (rgb_png.shape[1], rgb_png.shape[0]))
# Normalize to [0, 1]
rgb_norm = rgb_png / 255.0
ir_norm = ir_png / 255.0

# Convert IR to 3 channels
ir_3ch = np.stack((ir_norm,)*3, axis=-1)


# Weighted fusion: alpha controls RGB contribution
alpha = 0.6
fused = cv2.addWeighted(rgb_norm, alpha, ir_3ch, 1 - alpha, 0)


fused_img = np.uint8(fused * 255)

# Save image
cv2.imwrite('fused_output.jpg', cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))

# Show result
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(rgb_png)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ir_png, cmap='gray')
plt.title("IR Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(fused_img)
plt.title("Fused Image")
plt.axis('off')
plt.show()
