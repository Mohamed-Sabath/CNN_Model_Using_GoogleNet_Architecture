# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:29:49 2024

@author: user
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# Function to create an Inception module
def inception_module(x, filters):
    conv1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = layers.Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)
    conv5x5 = layers.Conv2D(filters[2], (5, 5), padding='same', activation='relu')(x)
    maxpool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_conv1x1 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(maxpool)

    return layers.concatenate([conv1x1, conv3x3, conv5x5, pool_conv1x1], axis=-1)

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# Define the Inception model with increased complexity
input_shape = (32, 32, 3)
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Increase model complexity
x = inception_module(x, [64, 128, 32, 32])
x = layers.Dropout(0.4)(x)
x = inception_module(x, [128, 256, 64, 64])
x = layers.MaxPooling2D((2, 2))(x)

# Additional dense layer
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.6)(x)

outputs = layers.Dense(10, activation='softmax')(x)

# Create the model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model with a learning rate scheduler
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model with data augmentation and learning rate scheduling
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=30,
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model to Google Drive
model.save('inception_model5.h5')

# Save training history to a text file on Google Drive
with open('inception_model_training_history.txt', 'w') as file:
    file.write(str(history.history))