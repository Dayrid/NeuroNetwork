import configparser
import json
import SQL
import restore_data
import pandas as pd
import numpy as np

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
        raw_data = restoring.raw_data
        print(raw_data)

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


a = Preprocessing()  # main1
