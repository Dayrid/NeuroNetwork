import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import main


class NeuroNetwork:
    data = main.Preprocessing()

    def __init__(self, filename=None):
        """
        Конструктор класса нейросети.
        :param filename: расположение файла с моделью нейросети
        """

        if filename is not None:
            self.model = tf.keras.models.load_model(filename)  # Создание модели через файл
        else:
            self.model = tf.keras.Sequential()  # Создание модели через конструктор

            #   Добавление входного слоя
            self.model.add(tf.keras.layers.Dense(30, input_shape=self.data.train_x.shape[1:], activation='linear'))

            # Добавление скрытых слоёв
            self.model.add(tf.keras.layers.SimpleRNN(150, activation='tanh', return_sequences=True))
            self.model.add(tf.keras.layers.SimpleRNN(50, activation='tanh'))

            # Выходной слой
            self.model.add(tf.keras.layers.Dense(5, activation='linear'))
        pass

    def fit(self):
        """
        Обучение нейросети
        """
        train_x, train_y = self.data.train_x, self.data.train_y  # Создание тренировочной выборки

        # Выбор функции ошибки и метода оптимизации градиентного спуска
        self.model.compile(loss='mse', optimizer='Adam', metrics='mae')

        # Обучение
        history = self.model.fit(train_x, train_y, epochs=200, batch_size=10, validation_split=0.2)

        pass

    def predict(self, data):
        return self.model.predict(data)

    def test(self):
        pass


