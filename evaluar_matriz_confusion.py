from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np

# 1. Cargar el modelo
model = load_model("action.h5")
