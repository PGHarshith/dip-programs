1.
from PIL import Image
import matplotlib.pyplot as plt

# Read the image
image_path = r"C:\Users\Admin\Desktop\test.jpg"
image = Image.open(image_paths)

# Get image dimensions
width, height = image.size
height=image.height 

# Calculate midpoints
mid_x, mid_y = width // 2, height // 2

# Split into four quadrants using the crop method
top_left = image.crop((0, 0, mid_x, mid_y))
top_right = image.crop((mid_x, 0, width, mid_y))
bottom_left = image.crop((0, mid_y, mid_x, height))
bottom_right = image.crop((mid_x, mid_y, width, height))

# Display the quadrants
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(top_left)
axes[0, 0].set_title("Top Left")
axes[0, 0].axis('off')

axes[0, 1].imshow(top_right)
axes[0, 1].set_title("Top Right")
axes[0, 1].axis('off')

axes[1, 0].imshow(bottom_left)
axes[1, 0].set_title("Bottom Left")
axes[1, 0].axis('off')

axes[1, 1].imshow(bottom_right)
axes[1, 1].set_title("Bottom Right")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


2.
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\subha\Desktop\download.jpeg"  # Update this path
image = Image.open(image_path)

# Rotation (Rotate by 45 degrees) 
rotated = image.rotate(45, expand=True)

# Scaling (Resize to 150% of original size) 
scale_factor = 1.5
new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
scaled = image.resize(new_size)

# Translation (Move image by 100px right, 50px down) 
translated = image.transform(image.size, Image.AFFINE, (1, 0, 100, 0, 1, 50))

# Display the original and transformed images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0,1].imshow(rotated)
axes[0,1].set_title("Rotated (45°)")
axes[0,1].axis("off")

axes[1,0].imshow(scaled)
axes[1,0].set_title("Scaled (150%)")
axes[1,0].axis("off")

axes[1,1].imshow(translated)
axes[1,1].set_title("Translated (100, 50)")
axes[1,1].axis("off")

plt.tight_layout()
plt.show()

3.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = r"C:\Users\subha\Desktop\download.jpeg"  # Update the path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the kernel for erosion
kernel = np.ones((5,5), np.uint8)  # 5x5 kernel

# Apply erosion
eroded = cv2.erode(image, kernel, iterations=1)

# Subtract the eroded image from the original
subtracted = cv2.subtract(image, eroded)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(eroded, cmap="gray")
axes[1].set_title("Eroded Image")
axes[1].axis("off")

axes[2].imshow(subtracted, cmap="gray")
axes[2].set_title("Subtracted (Original - Eroded)")
axes[2].axis("off")

plt.tight_layout()
plt.show()

4.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = r"C:\Users\subha\Desktop\download.jpeg"  # Update the path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel

# Apply erosion
eroded = cv2.erode(image, kernel, iterations=1)
edge_erosion = cv2.subtract(image, eroded)  # Edge by subtracting eroded image

# Apply dilation
dilated = cv2.dilate(image, kernel, iterations=1)
edge_dilation = cv2.subtract(dilated, image)  # Edge by subtracting original from dilated

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(edge_erosion, cmap="gray")
axes[1].set_title("Edge using Erosion")
axes[1].axis("off")

axes[2].imshow(edge_dilation, cmap="gray")
axes[2].set_title("Edge using Dilation")
axes[2].axis("off")

plt.tight_layout()
plt.show()

5.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a low-contrast grayscale image
image_path = r"C:\Users\subha\Desktop\download.jpeg"  # Update the path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Histogram Equalization 
equalized = cv2.equalizeHist(image)
# Image Segmentation using Otsu's Thresholding 
_, binary_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(12, 10))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Low-Contrast Image")
axes[0].axis("off")

axes[1].imshow(equalized, cmap="gray")
axes[1].set_title("Histogram Equalized")
axes[1].axis("off")

axes[2].imshow(binary_thresh, cmap="gray")
axes[2].set_title("Segmented (Otsu's Thresholding)")
axes[2].axis("off")

plt.tight_layout()
plt.show()

6.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\subha\Desktop\download.jpeg"  # Update your path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Laplacian filter
laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

# Apply Sobel filters in x and y directions
sobel_x_image = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y_image = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)

# Apply Canny edge detection
canny_image = cv2.Canny(blurred_image, 30, 100)

# Display the results using Matplotlib
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(blurred_image, cmap="gray")
axes[0, 1].set_title("Blurred Image")
axes[0, 1].axis("off")

axes[0, 2].imshow(laplacian_image, cmap="gray")
axes[0, 2].set_title("Laplacian Edge Detection")
axes[0, 2].axis("off")

axes[1, 0].imshow(sobel_x_image, cmap="gray")
axes[1, 0].set_title("Sobel X Edge Detection")
axes[1, 0].axis("off")

axes[1, 1].imshow(sobel_y_image, cmap="gray")
axes[1, 1].set_title("Sobel Y Edge Detection")
axes[1, 1].axis("off")

axes[1, 2].imshow(canny_image, cmap="gray")
axes[1, 2].set_title("Canny Edge Detection")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

7.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the RGB image
image_path = r"C:\Users\subha\Desktop\download.jpeg"   # Update with your image path
image = cv2.imread(image_path)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to Binary (Thresholding)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Display the images using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original RGB Image")
axes[0].axis("off")

axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Grayscale Image")
axes[1].axis("off")

axes[2].imshow(binary_image, cmap="gray")
axes[2].set_title("Binary Image (Black & White)")
axes[2].axis("off")

plt.tight_layout()
plt.show() 
