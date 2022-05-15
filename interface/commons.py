import os
import io
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

CLASSES = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

def transform_image(image_path) : 
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64,64))
    img_y = img.astype("float") / 255.0
    img_y = np.expand_dims(img_y, axis=0)
    return img_y

def format_class_name(conf):
    idx = np.argmax(conf)
    label = CLASSES[idx]
    return label

# Pass a PIL image, return a tensor
def scaleImage(x):          
    toTensor = tf.ToTensor()
    y = toTensor(x)
    if(y.min() < y.max()):  
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        
    return z

