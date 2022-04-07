from tensorflow import keras
from keras import layers
import numpy as np

# subclassing model


class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_clf = layers.Dense(num_departments, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_clf(features)
        return priority, department


# training model

vocab_size = 10000
num_tags = 100
num_dept = 4

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocab_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_dept))

model = CustomerTicketModel(num_departments=4)

model.compile(
    optimizer="rmsprop",
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]]
)

model.fit({
    "title": title_data,
    "text_body": text_body_data,
    "tags": tags_data
},
    [priority_data, department_data],
    epochs=10
)

model.evaluate({
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
},
    [priority_data, department_data]
)

priority_preds, department_preds = model.predict(

    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    }
)

print(f"Priority Prediction : {priority_preds}")
print(f"Department Predictions : {department_preds}")
