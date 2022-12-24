import numpy as np
import tensorflow as tf 
from tensorflow import keras

from keras.models import load_model



model = load_model('keras_model.h5')



def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
