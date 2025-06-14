from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

actions = np.array(["hola", "gracias", "te amo"])
label_map = {label: num for num, label in enumerate(actions)}
print(label_map)