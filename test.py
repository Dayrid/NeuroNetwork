import pandas as pd
import SQL
import restore_data
import configparser
import numpy as np


pd.options.display.max_columns = None
pd.options.display.max_rows = None


class TestingRestore:
    def __init__(self):
        self.params = self.config('Settings.ini', 'Settings')
        self.add_params = self.config('Settings.ini', 'Test')
        if self.params['sql_reading'].lower() == 'on':
            sql = SQL.SQL(self.params['hydropost'])
            not_restored_df = sql.df
        else:
            not_restored_df = self.xlsx_read(self.params['pathtofile'])
        # values = not_restored_df.loc[
        #          1:int(len(not_restored_df) * self.add_params['test_selection_size'])+1, 'Уровень воды'].tolist()
        test_size = int(len(not_restored_df) * self.add_params['test_selection_size'])+1
        test_values = [[1, test_size], [len(not_restored_df)//2 - test_size//2, len(not_restored_df)//2 + test_size//2], [len(not_restored_df)-test_size, len(not_restored_df)]]
        values = not_restored_df.loc[test_values[self.add_params['selection_pos']][0]:test_values[self.add_params['selection_pos']][1],'Уровень воды'].tolist()
        not_restored_df = not_restored_df.drop(range(1, test_size))
        not_restored_df = not_restored_df.reset_index(drop=True)
        restoring = restore_data.DataRestore(not_restored_df, self.params)
        self.raw_data = restoring.raw_data
        new_values = self.raw_data.loc[
                 test_values[self.add_params['selection_pos']][0]:test_values[self.add_params['selection_pos']][1], 'Уровень воды'].tolist()
        err = [abs(values[i] - new_values[i]) for i in range(len(values))]
        err = np.array(err)
        print(err.mean())

    def xlsx_read(self, filename):
        # Чтение из xlsx формата
        dfs = pd.read_excel(filename, sheet_name='Уровни', engine='openpyxl')
        dfs = dfs[dfs['Код поста'] == int(self.params['hydropost'])]
        dfs = dfs.sort_values('Дата - время')
        dfs = dfs.sort_values('Дата - время')
        return dfs

    @staticmethod
    def config(name, category):
        config = configparser.ConfigParser()
        config.read(name, encoding="utf-8")
        array = dict(config.items(category))
        for key in array.keys():
            if array[key].isdigit():
                array[key] = int(array[key])
            elif array[key].replace('.', '1').isdigit():
                array[key] = float(array[key])
            elif ',' in array[key]:
                array[key] = array[key].split(',')
                array[key] = [value for value in array[key] if value]
        return array


a = TestingRestore()
