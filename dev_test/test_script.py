import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# x = np.reshape(np.random.randint(0, 20, 24), (2, 3, 4))

# s = np.array([[1e-14, 1e-12, 0.0, 1e-6, 1e-22, 0.0], [1e-14, 1e-12, 0.0, 1e-6, 1e-22, 0.0]], dtype=np.float64)
# print('np.argwhere(s == 0)[0]', np.where(s == 0))
# where = np.where(s == 0)
# s[where] = np.nan
# print(s)

# s = np.reshape(np.arange(0, 20, 1), (5, 4))
# v = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
# v_prime = np.array([1, 2, 3, 4, 5])
# v_repeat = np.repeat(v, 4, axis=0)
# s_repeat = np.repeat(s, 4, axis=0)
# print(s_repeat * v_repeat)
# print(s, v)
# print(v*s[:, 0])
# print()

s = np.array([1+3j, 2+0j, -6+6j], dtype=np.complex)

print(s)
print(s/np.linalg.norm(s))
print(np.linalg.norm(s/np.linalg.norm(s)))


