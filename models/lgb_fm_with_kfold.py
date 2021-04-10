import pandas as pd
import numpy as np
import math
import time

import torch
import torch.nn as nn

import sys
sys.path.append('./src')

from lxyTools.singleModelUtils import singleModel
from lxyTools.adversarialValidation import adversarialValidation

from specialTools.model import lgb_nn_with_kfold
from specialTools.answer import Online_Metric
from specialTools.answer import test_df_to_answer

from lxyTools.singleModelUtils import singleModel
from lxyTools.adversarialValidation import adversarialValidation
from lxyTools.pytorchTools import set_random_seed
from lxyTools.pytorchTools import myBaseModule
from lxyTools.pytorchTools import Attention
from lxyTools.pytorchTools import BiInteraction
from lxyTools.pytorchTools import CrossNet
from lxyTools.pytorchTools import FM

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


from tqdm import tqdm
tqdm.pandas(desc="my bar!")

set_random_seed(1546)

# 读取数据
train = pd.read_csv('./processedData/train.csv')
train.create_time = pd.to_datetime(train.create_time)
train.days = pd.to_datetime(train.days)

valid = pd.read_csv('./processedData/valid.csv')
valid.create_time = pd.to_datetime(valid.create_time)
valid.days = pd.to_datetime(valid.days)

# 合并训练数据，以及验证数据，给模型足够的数据去训练
train = pd.concat([train, valid])
train = train.reset_index(drop=True)

# 对于为零的样本进行降采样
train_select_0 = train[train.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
train_select_1 = train[train.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
train_select = train_select_0 + train_select_1
train = train.loc[train_select]
train = train.sample(frac=1.0, random_state=111)


valid_select_0 = valid[valid.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
valid_select_1 = valid[valid.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
valid_select = valid_select_0 + valid_select_1
valid = valid.loc[valid_select]
valid = valid.sample(frac=1.0, random_state=111)

test = pd.read_csv('processedData/test.csv')
test.create_time = pd.to_datetime(test.create_time)

# 特征选择
features_names = ['item_category', 'ad_size_id', 'ad_account_id', 'item_id', 'cost_mode', 'object',
                  'requestion_num', 'rival_epcm', 'existingTime', 'rival_max_epcm', 'user_num', 'rival_pctr', 'rival_filter_rate', 'rival_q_epcm']

for col in train.columns:
    if ('_2day' in col or '_3day' in col):
        features_names.append(col)

# 标明某一些特征要作为类别特征去训练
cat_features = ['item_category', 'ad_size_id', 'ad_account_id', 'item_id', 'cost_mode', 'object']


train_x = train[features_names]
train_y = train.Y


valid_x = valid[features_names]
valid_y = valid.Y

test_x = test[features_names]


'''
    lgb模型的目标函数
'''

def special_Objective(y_true, y_pred):

    y_true = pd.Series(y_true).apply(lambda x: max(x, 0.1)).values
    y_pred = np.array(y_pred)

    err = y_pred - y_true
    err_sign = np.sign(err)

    err_abs = np.abs(err)
    delta = 150

    grad = np.zeros(err.shape[0])
    hess = np.zeros(err.shape[0])

    grad[err_abs<delta] = (2*err/((y_true)))[err_abs<delta]
    hess[err_abs<delta] = (2/((y_true)))[err_abs<delta]

    grad[err_abs>=delta] = (delta * err_sign/((y_true)))[err_abs>=delta]
    return grad, hess

'''
    lgb模型的参数
'''
num_leaves = 1024
num_trees = 600

lgb_param = {
    'boosting_type': 'gbdt',
    'objective': special_Objective,
    'num_leaves': num_leaves,
    'max_depth': -1,
    'learning_rate': 0.05,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': num_trees,
    'reg_alpha': 10,
}


'''
    fm模型的结构以及参数
'''
class nn_lr(nn.Module, myBaseModule):
    def __init__(self, random_seed, input_type=torch.long):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed, input_type=torch.long)

        self.dropout = nn.Dropout(0.1)
        # self.dnn_feature_extract = nn.Sequential(
        #             nn.Linear(num_trees*1, 128),
        #             nn.ReLU(),
        #             nn.Linear(128, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 256),
        #             nn.ReLU(),
        # )

        self.embedding_linear = nn.Linear(num_trees*num_leaves, 3, bias=False).cuda()
        nn.init.xavier_uniform_(self.embedding_linear.weight)
        self.linear = FM(num_trees*3, 1, bias=False).cuda()


        self.out_bias = torch.tensor(100, dtype=torch.float32).cuda()
        self.out_bias = nn.Parameter(self.out_bias)

        def loss_fn(x, y):

            x = torch.nn.functional.relu(x - 1) + 1
            y = torch.nn.functional.relu(y - 1) + 1

            re = torch.abs(x - y)/torch.abs(x + y)

            return torch.sum(re)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))

        self.vis = None

    def forward(self, x):

        x = nn.functional.embedding(x, self.embedding_linear.weight.t().contiguous())
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        linear_part = self.linear(x) + self.out_bias

        return linear_part.reshape(-1)

# 建立lgb+fm模型
set_random_seed(1546)
model = lgb_nn_with_kfold(lgb.LGBMRegressor, lgb_param, nn_lr, {})

# 训练模型
model.fit(train_x, train_y, cat_features, valid_x, valid_y, KFold(n_splits=5, random_state=0), 50, 2048)

# 读取测试数据
test = pd.read_csv('processedData/test.csv')
test.create_time = pd.to_datetime(test.create_time)

# lgb+fm模型预测线上结果
test['model_predict'] = model.predict(test[train_x.columns])
test[['test_id', 'model_predict']].to_csv('./processedData/lgb_fm_test_predict_val.csv', index=False)
