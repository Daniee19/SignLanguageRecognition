from keras.models import load_model
import numpy as np


class SignModel:
    def __init__(self, path="model/action.h5"):
        self.model = load_model(path)
        self.actions = ["hola", "gracias", "te amo"]
        self.threshold = 0.5
        self.sequence = []

    def predict(self, keypoints):
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        if len(self.sequence) < 30:
            return None, None
        res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
        idx = np.argmax(res)
        if res[idx] > self.threshold:
            return self.actions[idx], float(res[idx])
        return None, res[idx]
