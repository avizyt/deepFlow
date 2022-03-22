import numpy as np

X = np.random.random(32,).reshape(8,4)
y = np.random.random(32,).reshape(4,8)
z = np.dot(X, y)

print(X)
print(X.shape)

print("====================================")
print(y)
print(y.shape)
print(z.shape)