import pandas as pd
# from sqlalchemy import create_engine
import pymysql.cursors
import configparser


class SQL:
    def __init__(self):
        params = self.cfg('/home/floodrb/sites/floodrb.ugatu.su/neuro/db.ini')
        self.df = self.sql_to_df(params['db_name'], params['username'], params['password'], params['hostname'],
                                 int(params['port']), params['table'], params['charset'])

    def sql_to_df(self, dbname, user, password, hostname, port, table, charset):
        con = pymysql.connect(host=hostname, port=port, user=user, password=password,
                              database=dbname, charset=charset)
        self.con = con.cursor()
        qry = f"""SELECT * FROM {table}"""
        df = pd.read_sql(sql=qry, con=con)
        return df

    def cfg(self, filename):
        config = configparser.ConfigParser()
        config.read(filename, encoding="utf-8")
        data = dict(config.items('SQL'))
        return data
