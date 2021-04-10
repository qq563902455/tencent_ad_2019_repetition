import numpy as np
import gc
# from fastFM import als
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


class BoosterLmModel:
    def __init__(self, nmlist, featurelist, lm=LogisticRegression()):
        self.nonlinearModellist = nmlist
        self.featurelist = featurelist
        self.linearModel = lm
        self.onehot = OneHotEncoder(sparse=True, dtype=np.uint8)


    def fit(self, X, y, eval_set_param_name=None, **kwargs):
        leafXlist = []

        for i in range(len(self.featurelist)):
            if eval_set_param_name is not None:
                kwargs_temp = kwargs.copy()

                kTest_x = kwargs[eval_set_param_name][0]
                kTest_y = kwargs[eval_set_param_name][1]

                kwargs_temp[eval_set_param_name] = (kTest_x[self.featurelist[i]], kTest_y)
            else:
                kwargs_temp = kwargs
            self.nonlinearModellist[i].fit(X[self.featurelist[i]], y, **kwargs_temp)
            leafXlist.append(
                self.nonlinearModellist[i].apply(X[self.featurelist[i]])
                )

        leafX = np.hstack(leafXlist)

        self.onehot.fit(leafX)
        onhotX = self.onehot.transform(leafX)

        # onhotX = np.hstack([onhotX, X[self.featurelist[i]].values])

        print(onhotX.shape)
        self.linearModel.fit(onhotX, y)

        del onhotX
        del leafX
        del leafXlist
        gc.collect()

    def predict_proba(self, X):
        leafXlist = []
        for i in range(len(self.featurelist)):
            leafXlist.append(
                self.nonlinearModellist[i].apply(X[self.featurelist[i]]))

        leafX = np.hstack(leafXlist)
        onhotX = self.onehot.transform(leafX)

        # onhotX = np.hstack([onhotX, X[self.featurelist[i]].values])

        out = self.linearModel.predict_proba(onhotX)

        del leafX
        del onhotX
        del leafXlist
        gc.collect()

        return out

    def predict(self, X):
        leafXlist = []
        for i in range(len(self.featurelist)):
            leafXlist.append(
                self.nonlinearModellist[i].apply(X[self.featurelist[i]]))

        leafX = np.hstack(leafXlist)
        onhotX = self.onehot.transform(leafX)

        # onhotX = np.hstack([onhotX, X[self.featurelist[i]].values])

        out = self.linearModel.predict(onhotX)

        del leafX
        del onhotX
        del leafXlist
        gc.collect()

        return out


'''
class FM(als.FMClassification):
    def fit(self, X, y):
        y_labels = np.array(y, dtype=np.int8)
        y_labels[y <= 0] = -1
#        print(np.min(y_labels))
#        print(np.max(y_labels))
        als.FMClassification.fit(self, X, y_labels)
'''
