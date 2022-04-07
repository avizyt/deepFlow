from tensorflow import keras
from keras import layers, Sequential


model = Sequential()
# input shape must be for each sample not for the batch
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

print(model.summary())

