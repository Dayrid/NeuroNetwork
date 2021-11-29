import configparser
from math import sqrt

import pandas as pd
import SQL
import numpy as np
from sklearn.linear_model import LinearRegression
pd.options.display.max_columns = None


def xlsx_read(filename):
    # Чтение из xlsx формата
    dfs = pd.read_excel(filename, sheet_name='Уровни', engine='openpyxl')
    dfs = dfs[dfs['Код поста'] == int(params['hydropost'])]
    dfs = dfs.sort_values('Дата - время')
    return dfs


def config(name):
    config = configparser.ConfigParser()
    config.read(name, encoding="utf-8")
    array = dict(config.items('Settings'))
    return array


if __name__ == '__main__':
    params = config('Settings.ini')
    params['selectedcols'] = params['selectedcols'].split(',')
    params['selection_size'] = int(params['selection_size'])
    params['predict_size'] = int(params['predict_size'])
    params['test_selection_size'] = float(params['test_selection_size'])

    sql = SQL.SQL(params['hydropost'])
    df = sql.df
    df = df[params['selectedcols']]

    df = df['2010-06-01' < df['Дата - время']]
    df = df[df['Дата - время'] < '2010-08-30']
    print(df)
    x, y = df[['Температура воздуха','Скорость ветра']].values.tolist(), df['Уровень воды'].values.tolist()

    x_1 = np.array(x)
    y_1 = np.array(y)

    lr = LinearRegression().fit(x_1, y_1)
    coef = sqrt(lr.score(x_1, y_1))
    print(coef)

    # print(np.corrcoef(x, y))

