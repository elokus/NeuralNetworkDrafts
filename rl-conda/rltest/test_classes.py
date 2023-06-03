import numpy as np
b = np.arange(48).reshape(1, 48)
a = np.array([[1, 1, 1, 1, 1]])
print(a.shape)
print(b.shape)
c = np.append(a, b).reshape(53,1)

print(c.shape)
