import datetime
import pandas as pd
import numpy as np
import copy
from restore_methods import naiveBayes, imputers


class DataRestore:
    def __init__(self, df, params):
        df = self.cutting(df, params)
        print(len(df))
        new_df = df
        if params['restore_data'].lower() != 'off':
            missed_date_df = self.fill_missed_date(df, params)
            if params['restore_data'] == 'naive_bayes':
                new_df = self.naive_bayes_restoring(missed_date_df, params)
            elif params['restore_data'] in ['knn', 'iter', 'mean']:
                new_df = self.imputers_restoring(missed_date_df, params, params['restore_data'], 5)
        else:
            print('Восстановление данных отключено.')
        self.raw_data = new_df
        self.raw_data = self.dates_selection(self.raw_data)

    @staticmethod
    def cutting(df, params):
        end_date = params['end_date']
        df['Дата - время'] = pd.to_datetime(df['Дата - время'])
        if end_date:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            end_date = datetime.datetime(end_date.year, end_date.month, end_date.day, hour=23,
                                         minute=59, second=59)
            date_idx = df.index[df['Дата - время'] > end_date].tolist()
            df = df.drop(date_idx)
            print('Файл готов к обработке.')
            return df

    @staticmethod
    def dates_selection(df):
        for i, row in df.iterrows():
            if not 3 <= row['Дата - время'].month <= 5:
                df = df.drop(i)
        return df

    @staticmethod
    def fill_missed_date(df, params):
        if params['restore_data'] != 'Off':
            if params['merge_missing_dates'] == 'on':
                m1 = df['Дата - время'].min()
                m2 = df['Дата - время'].max()
                df = df.set_index('Дата - время')
                df = df.reindex(pd.date_range(m1, m2)).fillna(np.nan)
                df = df.reset_index()
                df.rename(columns={'index': 'Дата - время'}, inplace=True)
            else:
                df = df.set_index('Дата - время').fillna(np.nan)
                df = df.reset_index()
                df.rename(columns={'index': 'Дата - время'}, inplace=True)
            return df

    @staticmethod
    def naive_bayes_restoring(df, params):
        obj = naiveBayes.NaiveBayes(df, params)
        return obj.restored_df

    @staticmethod
    def imputers_restoring(df, params, method, k=None):
        obj = imputers.Imputers(df, params, method, k)
        return obj.restored_df
