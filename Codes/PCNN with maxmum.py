import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input images
image1 = cv2.imread("g1.png")
image2 = cv2.imread("g2.png")

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply the PCNN algorithm on the images
def PCNN(img):
    # Set the PCNN parameters
    tao = 0.4
    beta = 0.2
    alpha = 0.2
    delta = 0.3

    # Initialize the membrane potential and the output image
    V = np.zeros(img.shape, dtype=np.float32)
    U = np.zeros(img.shape, dtype=np.float32)

    # Set the mask
    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)

    # Iterate the PCNN algorithm
    for i in range(3):
        X = img * V
        Y = cv2.filter2D(V, -1, mask)
        V = np.exp(-tao) * V + beta * X - alpha * Y + delta

        # Calculate the output image
        U = U + V

    return U

# Apply the PCNN algorithm on both grayscale images
U1 = PCNN(gray1)
U2 = PCNN(gray2)

# Normalize the output images
U1_norm = cv2.normalize(U1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
U2_norm = cv2.normalize(U2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Perform image fusion using the max rule
fusion = cv2.max(U1_norm, U2_norm)

# Display the input and output images
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(U1_norm, cmap='gray')
plt.title('Output Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(U2_norm, cmap='gray')
plt.title('Output Image 2'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(fusion, cmap='gray')
plt.title('Fused Image'), plt.xticks([]), plt.yticks([])
plt.show()
