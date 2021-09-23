import configparser
import json
import SQL
import restore_data

class Preprocessing:
    def __init__(self):
        self.params = self.config('Settings.ini')
        self.params['selectedcols'] = self.params['selectedcols'].split(',')
        sql = SQL.SQL(self.params['hydropost'])
        df = sql.df
        restoring = restore_data.DataRestore(df, self.params)

    @staticmethod
    def config(name):
        config = configparser.ConfigParser()
        config.read(name, encoding="utf-8")
        array = dict(config.items('Settings'))
        return array


a = Preprocessing()
