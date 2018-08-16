import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.expand_dims(train_images, 3)
test_images = np.expand_dims(test_images, 3)

ff_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(1568, activation=tf.nn.relu),
    keras.layers.Dense(392, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
ff_model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])
print(ff_model.summary())
ff_model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = ff_model.evaluate(test_images, test_labels)
print('Feed forward model test accuracy:', test_acc)

print()
print()
print()

cnn_model = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(392, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])
cnn_model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])
print(cnn_model.summary())
print(train_images.shape)
cnn_model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('CNN model test accuracy:', test_acc)


