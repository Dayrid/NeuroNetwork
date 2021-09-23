import numpy as np
from sklearn.naive_bayes import GaussianNB
import datetime
import math
import copy


class NaiveBayes:
    def __init__(self, df, params):
        self.df = df
        self.cfg = params
        self.restored_df = self.naive_bayes_method(self.df)

    def naive_bayes_method(self, df):
        df = df.sort_values(by=['Дата - время'])
        df = df.reset_index(drop=True)
        df.to_excel('before_restore.xlsx')
        dates_list = df['Дата - время'].tolist()
        dates_list = [str(i).split(' ')[0] for i in dates_list]
        day_num_list = [int(datetime.datetime.strptime(i, "%Y-%m-%d").strftime("%j")) for i in dates_list]
        day_num_list_train = {}
        data_train = {}
        data_predicted = {}
        for col in self.cfg['selectedcols']:
            data_train[col] = []
            day_num_list_train[col] = []
            for i in range(0, len(df[col])):
                if not np.isnan(df[col][i]):
                    data_train[col].append(df[col][i])
                    day_num_list_train[col].append(day_num_list[i])
        for col in self.cfg['selectedcols']:
            x = np.array(day_num_list_train[col]).reshape(-1, 1)
            y = np.array(data_train[col])
            model = GaussianNB()
            model.fit(x, y)
            data_predicted[col] = model.predict(np.array(day_num_list).reshape(-1, 1)).tolist()
        for col in self.cfg['selectedcols']:
            for i in range(0, len(df[col])):
                if np.isnan(df[col][i]):
                    df.loc[i, col] = data_predicted[col][i]
                    df.loc[i, 'Код поста'] = int(self.cfg['hydropost'])
                    df.loc[i, 'Код параметра'] = 1
        print("Восстановление методом Наивного Байеса завершено.")
        # Выгрузка результатов восстановления в эксель
        df.to_excel('after_restore.xlsx', 'Naive Bayes')
        return df

    @staticmethod
    def data_check(date: str):
        sum = 0
        month_dict = {
            1: 31,
            2: 28,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
        date_list = date.split('-')
        for i in range(1, int(date_list[1])):
            sum += month_dict[i]
        sum += int(date_list[-1])
        return sum
