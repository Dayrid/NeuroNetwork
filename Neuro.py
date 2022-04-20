import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from main import Preprocessing
import datetime

import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
            # self.model.add(tf.keras.layers.SimpleRNN(80, activation='sigmoid', return_sequences=True))
            # self.model.add(tf.keras.layers.SimpleRNN(160, activation='sigmoid', return_sequences=True))
            self.model.add(tf.keras.layers.SimpleRNN(20, activation='sigmoid'))

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
        history = self.model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,
                                 validation_split=validation_split)

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
        index = [datetime.datetime.strptime(a, '%Y-%m-%d') for a in set(sum(data.test_y_dates, []))]
        df = pd.DataFrame(columns=['real', '1DayErr', "2DayErr", "3DayErr", '4DayErr', "5DayErr", '1DayPredict',
                                   "2DayPredict", "3DayPredict", '4DayPredict', "5DayPredict"],
                          index=index).sort_index()
        # df.loc['2021-04-12', 'real'] = 1
        history = self.model.evaluate(data.test_x, data.test_y)

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
            for j, row in df.iterrows():
                j = j.strftime("%Y-%m-%d")
                if j in data.test_y_dates[i]:
                    idx = data.test_y_dates[i].index(j)
                    df.loc[j, f'{idx+1}DayPredict'] = predict[0][idx]
                    df.loc[j, f'{idx+1}DayErr'] = abs(y[idx] - predict[0][idx])
                    df.loc[j, 'real'] = y[idx]
            plt.legend(['До', 'Реальные значения', 'Прогноз'])
            plt.xticks(rotation=45)
            # plt.show()
        df.loc['mean_value'] = df.mean()
        today = datetime.datetime.now()
        today = today.strftime("%Y-%m-%d_%H-%M-%S")
        with pd.ExcelWriter(f"Test {self.params['hydropost']} fd{self.params['selection_size']} fh{self.params['predict_size']} {today}.xlsx", engine='xlsxwriter') as wb:
            df.to_excel(wb, sheet_name='Results', float_format="%.2f")
            sheet = wb.sheets['Results']
            header_format = wb.book.add_format({'font_name':'Times New Roman', 'font_size': 10, 'bold':False,
                                              'font_color':'blue', 'bg_color':'#AAAAAA', 'border':1, 'align':'center'})
            rows_format = wb.book.add_format({'font_name':'Times New Roman', 'font_size': 10, 'bold':False,
                                              'font_color':'blue', 'bg_color':'#AAAAAA', 'border':1, 'align':'center'})
            cell_format = wb.book.add_format({'font_name':'Times New Roman', 'font_size': 10, 'bold':False,
                                              'font_color':'black', 'border':1, 'align':'center'})
            for col_num, value in enumerate(df.columns.values):
                sheet.write(0, col_num + 1, value, header_format)
            indexes = df.index.astype(str)
            for idx, value in enumerate(df.index.values):
                sheet.write(idx+1, 0, indexes[idx], rows_format)
            k = 0
            cols = df.columns.tolist()
            for i, row in df.iterrows():
                for col in cols:
                    if pd.isnull(row[col]):
                        sheet.write(k + 1, cols.index(col) + 1, 'NULL', cell_format)
                    else:
                        sheet.write(k+1, cols.index(col)+1, row[col], cell_format)
                k+=1
            sheet.set_column('A:A', 18)
            sheet.set_column('B:L', 11)
        print(mean_error / len(data.test_x))
        pass

    def save(self, name):
        self.model.save(name)
