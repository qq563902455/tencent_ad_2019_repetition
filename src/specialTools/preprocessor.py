import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class id_count_convert:
    def __init__(self, dataset_list=[], id_cols=[]):
        self.cat_map_dict = {}
        for col in id_cols:
            map_dict = pd.concat([dataset[col] for dataset in dataset_list]).value_counts(dropna=False)
            self.cat_map_dict[col] = map_dict

    def transform(self, X):
        re = np.zeros((X.shape[0], len(self.cat_map_dict.keys())))
        for i, col in enumerate(self.cat_map_dict.keys()):
            re[:, i] = X[col].map(self.cat_map_dict[col])
        return re.astype(int)

class continous_col_discretizer:
    def __init__(self, dataset_list=[], cols=[], n_bins=10):
        self.cols = cols
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
        cols_value_list = pd.concat([dataset[cols] for dataset in dataset_list])
        self.discretizer.fit(cols_value_list)
    def transform(self, X):
        X = X[self.cols]
        return self.discretizer.transform(X)

class id_index_converter:
    def __init__(self, dataset_list=[],
                 id_col=['item_category', 'ad_size_id', 'cost_mode']):

         self.cat_list_dict = {}

         for col in id_col:
             id_cat_list = pd.concat([dataset[col] for dataset in dataset_list]).unique().tolist()
             self.cat_list_dict[col] = id_cat_list

    def transform(self, X):
         re = np.zeros((X.shape[0], len(self.cat_list_dict.keys())))

         for i, col in enumerate(self.cat_list_dict.keys()):
             re[:, i] = X[col].apply(lambda x: self.cat_list_dict[col].index(x))

         return re.astype(int)

class continous_col_converter:
    def __init__(self, dataset_list=[],
                       cols=[], percentile=90):

        self.min_cols_dict = {}
        self.percentile_cols_dict = {}
        self.mean_cols_dict = {}

        for col in cols:
            pd_series = pd.concat([dataset[col] for dataset in dataset_list])
            if pd_series.std() > 1e-5:
                self.min_cols_dict[col] = pd_series.min()
                pd_series = pd_series - self.min_cols_dict[col]
                self.percentile_cols_dict[col] = np.percentile(pd_series, percentile)
                pd_series = pd_series/self.percentile_cols_dict[col]
                self.mean_cols_dict[col] = pd_series.mean()

    def transform(self, X):
        re = np.zeros((X.shape[0], len(self.min_cols_dict.keys())))
        for i, col in enumerate(self.min_cols_dict.keys()):
             re[:, i] = X[col] - self.min_cols_dict[col]
             re[:, i] = re[:, i] / self.percentile_cols_dict[col]
             re[:, i] = re[:, i] - self.mean_cols_dict[col]
        return re
