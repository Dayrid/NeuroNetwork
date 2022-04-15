import Neuro
# import main
import configparser
import pymysql
import json
import datetime
import numpy as np


def normalize(data, min_max):
    min_max = min_max.T
    min_arr = np.array([min(min_max[i]) for i in range(len(min_max))])
    max_arr = np.array([max(min_max[i]) for i in range(len(min_max))])
    data = (data - min_arr) / (max_arr - min_arr)
    return data


def denormalize(predict, min_max):
    min_max = min_max.T
    predict = predict*(min_max[0][1] - min_max[0][0]) + min_max[0][0]
    return predict


def Predict(json_settings):
    sql_cfg = cfg("db.ini")
    con = pymysql.connect(host=sql_cfg['hostname'], port=int(sql_cfg['port']), user=sql_cfg['username'],
                          password=sql_cfg['password'],
                          database=sql_cfg['db_name'], charset=sql_cfg['charset'])
    cur = con.cursor()
    json_settings = json.loads(json_settings)
    start_date = datetime.datetime.strptime(json_settings['end_date'], '%Y-%m-%d')
    start_date -= datetime.timedelta(days=int(json_settings['selection_size']) - 1)
    query = f"SELECT `Уровень воды`, `Температура воздуха` FROM `{sql_cfg['table']}` WHERE `Код поста` = {json_settings['hydropost']} AND `Дата - время` BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{json_settings['end_date']}'"
    cur.execute(query)
    data = np.array(cur.fetchall())
    print(data)
    query = f"SELECT MIN(`Уровень воды`), MIN(`Температура воздуха`) as min, MAX(`Уровень воды`), MAX(`Температура воздуха`) as max FROM `{sql_cfg['table']}` WHERE `Код поста` = {json_settings['hydropost']}"
    cur.execute(query)
    min_max = np.array(cur.fetchall())
    min_max = min_max.reshape((len(min_max[0])//2, 2))
    data = normalize(data, min_max)
    net = Neuro.NeuroNetwork(data.shape, json_settings["model_path"])
    predict = net.predict(data)
    predict = denormalize(predict, min_max)
    print(predict)
    full_predict = []
    if json_settings['id'] != -1:
        end_date = datetime.datetime.strptime(json_settings['end_date'], '%Y-%m-%d')
        for i in range(len(predict[0])):
            end_date += datetime.timedelta(days=1)
            full_predict.append([end_date.strftime('%Y-%m-%d'), predict[0][i]])
    print(full_predict)


def cfg(filename):
    config = configparser.ConfigParser()
    config.read(filename, encoding="utf-8")
    data = dict(config.items('SQL'))
    return data

import datetime as t

s = t.datetime.now()

json_data = """
{
	"hydropost" : 76289,
	"end_date" : "2021-04-15",
	"selection_size" : 6,
	"predict_size" : 5,
	"id" : 123,
	"model_path" : "76289-5.h5"
}
"""
Predict(json_data)


print(t.datetime.now() - s)