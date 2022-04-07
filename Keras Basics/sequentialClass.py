from tensorflow import keras
from keras import layers

# list type of model
model = keras.Sequential([
    layers.Dense(64, activation="relu", name="1st_layer"),
    layers.Dense(10, activation="softmax", name="2nd_layer")
])

# model using built-in add() method
model_2 = keras.Sequential()
model_2.add(layers.Dense(64, activation="relu", name="1st_layer"))
model_2.add(layers.Dense(10, activation="softmax", name="2nd_layer"))


model_2.build(input_shape=(None, 3))

# print(model.weights)
print(model_2.summary())
