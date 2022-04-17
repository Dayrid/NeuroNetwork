import pandas as pd
# from sqlalchemy import create_engine
import pymysql.cursors
import configparser


class SQL:
    def __init__(self, hydropost):
        params = self.cfg('db.ini')
        self.df = self.get_main_df(params['db_name'], params['username'], params['password'], params['hostname'],
                                 int(params['port']), params['table'], params['charset'], hydropost)

    def get_main_df(self, dbname, user, password, hostname, port, table, charset, hydropost):
        con = pymysql.connect(host=hostname, port=port, user=user, password=password,
                              database=dbname, charset=charset)
        self.con = con.cursor()
        qry = f"""SELECT * FROM {table} WHERE `Код поста` = {hydropost}"""
        df = pd.read_sql(sql=qry, con=con)
        return df

    def cfg(self, filename):
        config = configparser.ConfigParser()
        config.read(filename, encoding="utf-8")
        data = dict(config.items('SQL'))
        return data
