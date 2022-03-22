import numpy as np

X = np.random.random((32,10))

print(X)
print(f"Shape of matrix X: {X.shape}")
print("=====================================")
y = np.random.random((10,))

print(y)
print(f" Shape of vector y: {y.shape}")
print("=====================================")
# now to make in broadcasting

y = np.expand_dims(y, axis=0)
y = np.concatenate([y] * 32, axis=0)

print(f" Shape of broadcast y: {y.shape}")
print(y)
print("=====================================")

z = np.add(X,y)

print(f" Shape of z: {z.shape}")
# print(z)
