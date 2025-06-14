import numpy as np
import os

path = os.path.join("MP_Data", "gracias", "0", "0.npy")
data = np.load(path)
print(data)
