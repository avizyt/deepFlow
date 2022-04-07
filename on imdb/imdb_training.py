import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.datasets import imdb

(train_data, train_labels), _ = imdb.load_data(num_words=10000)


def vectorize_sequence(seqs, dim=10000):
    res = np.zeros((len(seqs), dim))
    for i , seq in enumerate(seqs):
        res[i, seq] = 1.
    return res


train_data = vectorize_sequence(train_data)


model = keras.Sequential([
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(80, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(60, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(40, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.RMSprop(1e-2)
loss = "binary_crossentropy"
metric = ["accuracy"]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metric
)

history_original = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=32,
    validation_split=0.4
)

train_loss = history_original.history['loss']
val_loss = history_original.history['val_loss']
epochs = range(1, 21)


# fig.add_trace(go.line(x=epochs, y=val_loss, labels={'x': 'epochs', 'y': 'val loss'}))
# fig.add_trace(go.line(x=epochs, y=train_loss, labels={'x': 'epochs', 'y': 'train loss'}))
fig1 = px.line(x=epochs, y=val_loss, labels={'x': 'epochs', 'y': 'val loss'})
fig2 = px.line(x=epochs, y=train_loss, labels={'x': 'epochs', 'y': 'train loss'})
fig1.show()
fig2.show()
