Program 
Deep learning based

from google.colab import drive
drive.mount('/content/drive')


!pip install opencv-python matplotlib numpy


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB and IR images (grayscale IR)
rgb_img = cv2.imread('/f6.png')         # RGB Image
ir_img = cv2.imread('/f7.png', 0)        # IR Image as grayscale

# Resize IR image to match RGB if needed
ir_resized = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

# Normalize IR image
ir_norm = cv2.normalize(ir_resized, None, 0, 255, cv2.NORM_MINMAX)


from google.colab import drive
drive.mount('/content/drive')

# Convert RGB to YCrCb and replace Y with IR image (intensity replacement)
ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = ir_norm
fused_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Save output
cv2.imwrite('/content/fused_simple.jpg', fused_img)

# Show
plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
plt.title("Simple Fused Image")
plt.axis('off')
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Sigmoid to normalize
        return x

# Convert images to tensor
rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
ir_tensor = torch.from_numpy(ir_norm).unsqueeze(0).unsqueeze(0).float() / 255.0

# Concatenate IR and RGB
fused_input = torch.cat((rgb_tensor, ir_tensor), dim=1)


model = FusionNet()
with torch.no_grad():
    fused_output = model(fused_input)

# Convert back to image
fused_img_dl = fused_output.squeeze().permute(1, 2, 0).numpy()
fused_img_dl = (fused_img_dl * 255).astype(np.uint8)

# Save and show
cv2.imwrite('/content/fused_dl.jpg', fused_img_dl)
plt.imshow(cv2.cvtColor(fused_img_dl, cv2.COLOR_BGR2RGB))
plt.title("Fused Image - Deep Learning")
plt.axis('off')
plt.show()
