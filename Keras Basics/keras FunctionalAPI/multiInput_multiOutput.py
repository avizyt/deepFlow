from tensorflow import keras
from keras import layers
import numpy as np


vocab_size = 10000
num_tags = 100
num_dept = 4

title = keras.Input(shape=(vocab_size,), name="title")
text_body = keras.Input(shape=(vocab_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_dept, activation="softmax", name="department"
)(features)

model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department]
)

print(model.summary())

# training model

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_dept))

model.compile(
    optimizer="rmsprop",
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]]
)

model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=10
)

model.evaluate(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data]
)

priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)

print(f"Priority Prediction : {priority_preds}")
print(f"Department Predictions : {department_preds}")


# keras.utils.plot_model(model, "ticket_classifier.png", show_shapes=True)
