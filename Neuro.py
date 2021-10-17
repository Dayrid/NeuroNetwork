import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import main


class NeuroNetwork:

    data = main.Preprocessing()

    def __init__(self, filename):
        self.model = tf.keras.models.load_model(filename)
        pass

    def __int__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(30, input_shape=(5, 2), activation='linear'))
        self.model.add(tf.keras.layers.SimpleRNN(150, activation='tanh'))
        self.model.add(tf.keras.layers.SimpleRNN(50, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(5, activation='linear'))
        pass

    def fit(self):
        pass

    def predict(self):
        pass
