import configparser
import json
import SQL
import restore_data
import pandas as pd
import numpy as np
import copy

pd.options.display.max_columns = None
pd.options.display.max_rows = None


class Preprocessing:
    def __init__(self):
        self.params = self.config('Settings.ini')
        self.params['selectedcols'] = self.params['selectedcols'].split(',')
        self.params['selection_size'] = int(self.params['selection_size'])
        self.params['predict_size'] = int(self.params['predict_size'])
        if self.params['sql_reading'].lower() == 'on':
            sql = SQL.SQL(self.params['hydropost'])
            df = sql.df
        else:
            df = self.xlsx_read(self.params['pathtofile'])
        restoring = restore_data.DataRestore(df, self.params)
        self.raw_data = restoring.raw_data
        # print(self.raw_data.tail(11))
        self.min_max = []
        self.train_x, self.x_full_data, self.train_y, self.y_full_data = self.cube_formation()

    def xlsx_read(self, filename):
        # Чтение из xlsx формата
        dfs = pd.read_excel(filename, sheet_name='Уровни', engine='openpyxl')
        dfs = dfs[dfs['Код поста'] == int(self.params['hydropost'])]
        dfs = dfs.sort_values('Дата - время')
        return dfs

    @staticmethod
    def config(name):
        config = configparser.ConfigParser()
        config.read(name, encoding="utf-8")
        array = dict(config.items('Settings'))
        return array

    def normalize(self):
        max = np.array(self.raw_data[self.params['selectedcols']].max().values.tolist())
        min = np.array(self.raw_data[self.params['selectedcols']].min().values.tolist())
        print(min, max)
        self.raw_data[self.params['selectedcols']] = (self.raw_data[self.params['selectedcols']] - min) / (max - min)
        self.min_max += [min, max]

    def denormalize(self):
        self.raw_data[self.params['selectedcols']] = self.raw_data[self.params['selectedcols']] * (
                self.min_max[1] - self.min_max[0]) + self.min_max[0]

    def cube_formation(self):
        """
        Массивы выборки (массив x или multiple в прошлом PM):
        selection_data: готовые данные выборки в numpy
        selection_dates: массив дат идентичный массиву selection_data
        selection_full_data: массив сшитых дат и данных
        """
        selection_data = []
        selection_dates = []
        selection_full_data = []
        """
        Массивы прогнозов (массив y или single в прошлом PM):
        predict_data: готовые данные прогнозов в numpy
        predict_dates: массив дат идентичный массиву predict_data
        predict_full_data: массив сшитых дат и данных
        """
        predict_data = []
        predict_dates = []
        predict_full_data = []

        df = self.raw_data
        columns = self.params['selectedcols']

        for i in range(len(df) - self.params['predict_size'] - self.params['selection_size'] + 1):
            selection_dates.append(df.loc[i:i+self.params['selection_size']-1, 'Дата - время'].astype(str).values.tolist())
            selection_data.append(df.loc[i:i+self.params['selection_size']-1, columns].values.tolist())
            selection_full_data.append([[selection_dates[-1][j], selection_data[-1][j]] for j in range(self.params['selection_size'])])
        selection_data = np.array(selection_data)
        # print(selection_data)

        for i in range(self.params['selection_size'], len(df) - self.params['predict_size'] + 1):
            predict_dates.append(df.loc[i:i+self.params['predict_size']-1, 'Дата - время'].astype(str).values.tolist())
            predict_data.append(df.loc[i:i+self.params['predict_size']-1, columns[0]].values.tolist())
            predict_full_data.append([[predict_dates[-1][j], predict_data[-1][j]] for j in range(self.params['predict_size'])])
        predict_data = np.array(predict_data)
        # print(predict_data)

        # for i in range(len(selection_full_data)):
        #     print(50*"-")
        #     print(selection_full_data[i])
        #     print(predict_full_data[i])
        return selection_data, selection_full_data, predict_data, predict_full_data


a = Preprocessing() # main1
