import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from main import Preprocessing
import os
import pandas as pd


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

    def __init__(self, shape, filename=None):
        """
        Конструктор класса нейросети.
        :param filename: расположение файла с моделью нейросети
        """
        self.params = Preprocessing.config("Settings.ini")
        if filename is not None:
            self.model = tf.keras.models.load_model(filename)  # Создание модели через файл
        else:
            self.model = tf.keras.Sequential()  # Создание модели через конструктор

            #   Добавление входного слоя
            self.model.add(tf.keras.layers.Dense(30, input_shape=shape, activation='linear'))

            # Добавление скрытых слоёв
            self.model.add(tf.keras.layers.SimpleRNN(80, activation='sigmoid', return_sequences=True))
            self.model.add(tf.keras.layers.SimpleRNN(160, activation='sigmoid', return_sequences=True))
            self.model.add(tf.keras.layers.SimpleRNN(80, activation='sigmoid'))

            # Выходной слой
            self.model.add(tf.keras.layers.Dense(5, activation='linear'))
        pass

    def fit(self, data, epoch, batch_size, validation_split, optimizer):
        """
        Обучение нейросети
        """

        train_x, train_y = data.train_x, data.train_y  # Создание тренировочной выборки

        # Выбор функции ошибки и метода оптимизации градиентного спуска
        self.model.compile(loss='mse', optimizer=optimizer, metrics='mae')

        # Обучение
        history = self.model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_split=validation_split)

        # Вывод графика процесса обучения
        plot_train_history(history, "Процесс обучения")
        pass

    def predict(self, data):
        data = np.array(data).reshape((1, self.params['selection_size'], len(self.params['selectedcols'])))
        return self.model.predict(data)

    @staticmethod
    def denormalize(predict, min_max):
        min_max = np.array(min_max).T
        predict = predict * (min_max[0][1] - min_max[0][0]) + min_max[0][0]
        return predict

    def test(self, data):
        history = self.model.evaluate(data.test_x, data.test_y)
        print(history)

        mean_error = np.array([0.0 for i in range(len(data.test_y[0]))])
        print(mean_error.shape)

        print(mean_error)

        for i in range(len(data.test_x)):
            x = data.test_x[i]
            y = data.test_y[i]
            predict = self.denormalize(self.predict(x), data.min_max)
            y = self.denormalize(y, data.min_max)
            print("Predict:")
            print(predict[0])
            print("Real")
            print(y)
            print("Error:")
            print(y, predict[0])
            print(y - predict[0], end="\n==================\n")
            mean_error += abs(y - predict[0])
            x = self.denormalize(x, data.min_max).T[0]
            plt.plot(data.test_x_dates[i], x)

            plt.plot(data.test_y_dates[i], y)
            plt.plot(data.test_y_dates[i], predict[0])

            plt.legend(['До', 'Реальные значения', 'Прогноз'])
            plt.xticks(rotation=45)
            # plt.show()

        print(mean_error / len(data.test_x))
        pass

    def save(self, name):
        self.model.save(name)



