name = "vision"

from .utils import subfolder_count
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.applications as models
from tensorflow.keras import backend as K
import scipy


def cnn_learner(data:ImageDataBunch, model:'model name'=models.resnet50.ResNet50):
    "Creates a Learner object"
    
    learn = Learner(data, model)
    learn.convNet()
    return learn