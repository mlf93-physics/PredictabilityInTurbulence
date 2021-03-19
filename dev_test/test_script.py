import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# x = np.reshape(np.random.randint(0, 20, 24), (2, 3, 4))

s = np.array([[1e-14, 1e-12, 0.0, 1e-6, 1e-22, 0.0], [1e-14, 1e-12, 0.0, 1e-6, 1e-22, 0.0]], dtype=np.float64)
print('np.argwhere(s == 0)[0]', np.where(s == 0))
where = np.where(s == 0)
s[where] = np.nan
print(s)