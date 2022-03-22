import numpy as np
from keras.activations import relu

X = np.random.random((20, 10))

W = np.random.random((10, 16))

b = np.random.random((16,))

W_dotX = np.dot(X, W)

print(W_dotX.shape)

y = W_dotX + b

print(y.shape)

# apply non-linear operation using relu()

output = relu(y)
print(output.shape)

