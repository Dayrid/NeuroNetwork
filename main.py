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
        if self.params['sql_reading'].lower() == 'on':
            sql = SQL.SQL(self.params['hydropost'])
            df = sql.df
        else:
            df = self.xlsx_read(self.params['pathtofile'])
        restoring = restore_data.DataRestore(df, self.params)
        self.raw_data = restoring.raw_data
        self.min_max = []

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
        pass


a = Preprocessing()  # main1
