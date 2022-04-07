import numpy as np
import pandas as pd
import plotly.express as px

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.datasets import mnist


# loading and reshaping data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_labels = train_labels.astype("float32") / 255

# building model
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compiling model
optimizer = keras.optimizers.RMSprop(1e-2)
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

history_model1 = model.fit(
    train_images,
    train_labels,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

val_loss = history_model1.history['val_loss']
epochs = range(1, 21)
fig = px.line(x=epochs, y=val_loss, labels={'x': 'epochs', 'y': 'val loss'})
fig.show()
