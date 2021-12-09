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

    def __init__(self, shape, filename=None):
        """
        Конструктор класса нейросети.
        :param filename: расположение файла с моделью нейросети
        """

        if filename is not None:
            self.model = tf.keras.models.load_model(filename)  # Создание модели через файл
        else:
            self.model = tf.keras.Sequential()  # Создание модели через конструктор

            #   Добавление входного слоя
            self.model.add(tf.keras.layers.Dense(30, input_shape=shape, activation='linear'))

            # Добавление скрытых слоёв
            self.model.add(tf.keras.layers.SimpleRNN(150, activation='tanh', return_sequences=True))
            self.model.add(tf.keras.layers.SimpleRNN(50, activation='tanh'))

            # Выходной слой
            self.model.add(tf.keras.layers.Dense(5, activation='linear'))
        pass

    def fit(self, data):
        """
        Обучение нейросети
        """

        train_x, train_y = data.train_x, data.train_y  # Создание тренировочной выборки

        # Выбор функции ошибки и метода оптимизации градиентного спуска
        self.model.compile(loss='mse', optimizer='Adam', metrics='mae')

        # Обучение
        history = self.model.fit(train_x, train_y, epochs=75, batch_size=10, validation_split=0.2)

        # Вывод графика процесса обучения
        plot_train_history(history, "Процесс обучения")
        pass

    def predict(self, data):
        data = np.array(data).reshape((1, 6, 2))
        return self.model.predict(data)

    def test(self, test_x, test_y):
        history = self.model.evaluate(test_x, test_y)
        print(history)

        mean_error = np.array([0.0 for i in range(len(test_y[0]))])
        print(mean_error.shape)

        print(mean_error)

        for i in range(len(test_x)):
            x = test_x[i]
            y = test_y[i]
            predict = self.predict(x)
            print("Predict:")
            print(predict[0])
            print("Real")
            print(y)
            print("Error:")
            print(y - predict[0], end="\n==================\n")
            mean_error += abs(y - predict[0])

        print(mean_error / len(test_x))
        pass
    def save(self, name):
        self.model.save(name)


data = main.Preprocessing()
net = NeuroNetwork(data.train_x.shape[1:])
net.fit(data)
net.save('76289-5.h5')
