import configparser
import json
import SQL


class Preprocessing:
    def __init__(self):
        self.params = self.config('Settings.ini')
        sql = SQL.SQL()
        df = sql.df
        print(df)

    @staticmethod
    def config(name):
        config = configparser.ConfigParser()
        config.read(name, encoding="utf-8")
        array = dict(config.items('Settings'))
        return array


a = Preprocessing()
