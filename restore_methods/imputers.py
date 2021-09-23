import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class Imputers:
    def __init__(self, df, params, method, k=None):
        self.df = df
        self.cfg = params
        self.k = 5 if k is None else k
        self.restored_df = self.impute(method)

    def impute(self, method):
        imputed_data = None
        cutted_df = self.df[self.cfg['selectedcols']]
        data_for_restore = np.array(cutted_df.values.tolist())
        if method == 'knn':
            model = KNNImputer(n_neighbors=self.k, missing_values=np.nan)
            imputed_data = model.fit_transform(data_for_restore)
        elif method == 'mean':
            model = SimpleImputer(missing_values=np.nan, strategy='median', verbose=0)
            imputed_data = model.fit(data_for_restore).transform(data_for_restore)
        elif method == 'iter':
            model = IterativeImputer(random_state=0, initial_strategy='median', missing_values=np.nan)
            imputed_data = model.fit_transform(data_for_restore)
        restored_data = imputed_data.T.tolist()
        for col in self.cfg['selectedcols']:
            self.df[col] = restored_data[self.cfg['selectedcols'].index(col)]
        self.df['Код поста'] = int(self.cfg['hydropost'])
        self.df['Код параметра'] = 1
        print(f"Восстановление методом {method} завершено.")
        # Выгрузка результатов восстановления в эксель
        self.df.to_excel('after_restore_' + method + '.xlsx', method)
        return self.df
