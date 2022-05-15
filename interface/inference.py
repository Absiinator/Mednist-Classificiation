import tensorflow as tf
import numpy as np
import os
import pickle
from commons import *

CLASS_NAME = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

def get_prediction(model, image_bytes):
    """
    Get the prediction of the model for the given image
    """
    
    # Get the prediction
    prediction = model.predict(image_bytes)
    # Get the class name from the prediction
    class_name = format_class_name(prediction)
    # Get the class id from the prediction
    class_id = np.argmax(prediction)
    return class_name, class_id