'''

The standard workflow of a deep learning model is to:
1. Define the model [classifier, regressor, etc.]
2. Compile the model [compile()]
3. Fit the model [fit()]
4. Evaluate the model [evaluate()]
5. Predict [predict()]

The Keras API provides a high-level interface to all of these steps.

'''

from keras import layers
from keras.datasets import mnist
from keras.layers import Dropout
from tensorflow import keras

import plotly.express as px


def get_mnist_model():
    inputs = keras.Input(shape=(28*28,))
    features = layers.Dense(512, activation='relu')(inputs)
    features = Dropout(0.2)(features)
    outputs = layers.Dense(10, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28*28)).astype("float32") / 255

train_images, val_images = train_images[10000:], train_images[:10000]
train_labels, val_labels = train_labels[10000:], train_labels[:10000]

model = get_mnist_model()
model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history_model = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))

test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)

train_loss = history_model.history['loss']
epochs = range(1, 21)
fig2 = px.line(x=epochs, y=train_loss, labels={'x': 'epochs', 'y': 'train loss'})
fig2.show()


val_loss = history_model.history['val_loss']
epochs = range(1, 21)
fig1 = px.line(x=epochs, y=val_loss, labels={'x': 'epochs', 'y': 'val loss'})
fig1.show()


