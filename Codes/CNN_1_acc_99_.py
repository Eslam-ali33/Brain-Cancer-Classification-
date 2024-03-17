"""
    New code
"""

import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import cv2
import os
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the input data and normalize it
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode the target labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# Define the CNN model
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the CNN model
model = cnn_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=200, verbose=2)

# Evaluate the trained model on the test dataset
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Load the input images and preprocess them
image1 = cv2.imread('D:/final project/_End-work/A123.jpg')
image1 = cv2.resize(image1, (256, 256))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = np.array(image1).reshape(1, 256, 256, 1).astype('float32') / 255

image2 = cv2.imread('D:/final project/_End-work/PETT.png')
image2 = cv2.resize(image2, (256, 256))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = np.array(image2).reshape(1, 256, 256, 1).astype('float32') / 255

# Fuse the two images
fused_image = (image1 + image2) / 2

# Get accuracy history
accuracy_history = history.history['loss']

# Display the fused image
plt.imshow(fused_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()


# Plot accuracy as a line graph
plt.plot(accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()