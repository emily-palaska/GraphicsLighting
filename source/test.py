import numpy as np

a = np.array([[2, 0, 0],
              [1, 2, 0],
              [0, 0, 2]])

b = np.array([[1], [0], [0]])
c = a - b
c = c / np.linalg.norm(c, axis=0)
print(np.sum(a, axis=0))