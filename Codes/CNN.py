import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from keras.optimizers import Adam

#Load medical images

image1 = cv2.imread('c2.png')
image2 = cv2.imread('c1.png')

#Preprocess the images

image1 = cv2.resize(image1, (256, 256))
image1 = image1.astype('float32') / 255.0
image2 = cv2.resize(image2, (256, 256))
image2 = image2.astype('float32') / 255.0

#Create a CNN model

input1 = layers.Input(shape=(256, 256, 3))
x1 = layers.Conv2D(16, 3, activation='relu')(input1)
x1 = layers.MaxPooling2D(2)(x1)
x1 = layers.Conv2D(32, 3, activation='relu')(x1)
x1 = layers.MaxPooling2D(2)(x1)
x1 = layers.Conv2D(64, 3, activation='relu')(x1)
x1 = layers.Flatten()(x1)

input2 = layers.Input(shape=(256, 256, 3))
x2 = layers.Conv2D(16, 3, activation='relu')(input2)
x2 = layers.MaxPooling2D(2)(x2)
x2 = layers.Conv2D(32, 3, activation='relu')(x2)
x2 = layers.MaxPooling2D(2)(x2)
x2 = layers.Conv2D(64, 3, activation='relu')(x2)
x2 = layers.Flatten()(x2)

merged = layers.concatenate([x1, x2])
x = layers.Dense(256, activation='relu')(merged)
x = layers.Dropout(0.5)(x)
output = layers.Dense(256*256*3, activation='sigmoid')(x)

optimizer = Adam(lr=1e-4)
model = models.Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')

#Train the model

model.fit([image1.reshape(1, 256, 256, 3), image2.reshape(1, 256, 256, 3)], 
          image1.reshape(1, 256*256*3), epochs=10, batch_size=32)



#Perform image fusion

output = model.predict([image1.reshape(1, 256, 256, 3), image2.reshape(1, 256, 256, 3)])
output = output.reshape(1, 256, 256, 3)
output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Evaluate the performance using PSNR and SSIM
psnr_value = psnr(image1, output[0, :, :, 0:3], data_range=output.max()-output.min())
ssim_value = ssim(image1, output[0, :, :, 0:3], multichannel=True, data_range=output.max()-output.min())

# Display the input images and the fused image
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Fused Image", output[0, :, :, 0:3])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the evaluation metrics
print("PSNR: {:.2f}".format(psnr_value))
print("SSIM: {:.2f}".format(ssim_value))


cv2.imwrite('CNN Fusion.png', output)
