import Neuro
# import main
import configparser
import pymysql
import json
import datetime
import numpy as np


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
    print(data.shape)
    net = Neuro.NeuroNetwork(data.shape)
    predict = net.predict(data)

    print(predict)


def cfg(filename):
    config = configparser.ConfigParser()
    config.read(filename, encoding="utf-8")
    data = dict(config.items('SQL'))
    return data


json_data = """
{
	"hydropost" : 76289,
	"end_date" : "2020-05-03",
	"selection_size" : 6,
	"predict_size" : 5,
	"id" : 123,
	"model_path" : "//"
}
"""
Predict(json_data)
