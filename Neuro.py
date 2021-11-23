import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import main


def plot_train_history(history, title):
    loss = history.history['mae']
    val_loss = history.history['val_mae']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


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
        history = self.model.fit(train_x, train_y, epochs=2, batch_size=10, validation_split=0.2)

        # Вывод графика процесса обучения
        plot_train_history(history, "Процесс обучения")
        pass

    def predict(self, data):
        data = np.array(data).reshape((1, 6, 2))
        return self.model.predict(data)

    def test(self):
        history = self.model.evaluate(self.data.test_x, self.data.test_y)
        print(history)
        for i in range(len(self.data.test_x)):
            x = self.data.test_x[i]
            y = self.data.test_y[i]
            predict = self.predict(x)
            print("Predict:")
            print(predict)
            print("Real")
            print(y)
            print("Error:")
            print(y - predict, end="\n==================\n")
        pass


net = NeuroNetwork()
net.fit()
net.test()
