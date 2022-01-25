import configparser
from math import sqrt

import matplotlib.pyplot as plt
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

    dfs = df['2010-03-01' < df['Дата - время']]
    dfs = dfs[dfs['Дата - время'] < '2010-05-01']

    df2 = df['2011-03-01' < df['Дата - время']]
    df2 = df2[df2['Дата - время'] < '2011-05-01']

    df3 = df['2012-03-01' < df['Дата - время']]
    df3 = df3[df3['Дата - время'] < '2012-05-01']

    df4 = df['2013-03-01' < df['Дата - время']]
    df4 = df4[df4['Дата - время'] < '2013-05-01']

    x, y = dfs[['Скорость ветра']].values.tolist(), dfs['Уровень воды'].values.tolist()
    x2, y2 = df2[['Скорость ветра']].values.tolist(), df2['Уровень воды'].values.tolist()
    x3, y3 = df3[['Скорость ветра']].values.tolist(), df3['Уровень воды'].values.tolist()

    x = np.array(x)
    y = np.array(y)

    x2 = np.array(x2)
    y2 = np.array(y2)

    x3 = np.array(x3)
    y3 = np.array(y3)

    x = np.vstack((x, x2, x3))
    y = np.hstack((y, y2, y3))
    print(x)
    print(y)
    lr = LinearRegression().fit(x, y)
    coef = sqrt(lr.score(x, y))
    print(coef)
    #plt.plot(x_1, y)
    #plt.show()
    # print(np.corrcoef(x, y))

