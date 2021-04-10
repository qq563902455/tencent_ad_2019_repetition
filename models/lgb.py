import pandas as pd
import numpy as np

import sys
sys.path.append('./src')

from lxyTools.singleModelUtils import singleModel
from lxyTools.adversarialValidation import adversarialValidation

from specialTools.model import classificationRegressor
from specialTools.answer import Online_Metric
from specialTools.answer import test_df_to_answer

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from tqdm import tqdm
tqdm.pandas(desc="my bar!")

# 读取数据
train = pd.read_csv('./processedData/train.csv')
train.create_time = pd.to_datetime(train.create_time)
train.days = pd.to_datetime(train.days)

# 对曝光量为0的样本进行降采样
train_select_0 = train[train.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
train_select_1 = train[train.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
train_select = train_select_0 + train_select_1
train = train.loc[train_select]
train = train.sample(frac=1.0, random_state=111)

# 读取数据
valid = pd.read_csv('./processedData/valid.csv')
valid.create_time = pd.to_datetime(valid.create_time)
valid.days = pd.to_datetime(valid.days)

# 对曝光量为0的样本进行降采样
valid_select_0 = valid[valid.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
valid_select_1 = valid[valid.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
valid_select = valid_select_0 + valid_select_1
valid = valid.loc[valid_select]
valid = valid.sample(frac=1.0, random_state=111)

# 特征选择
features_names = ['item_category', 'ad_size_id', 'ad_account_id', 'item_id', 'cost_mode', 'object',
                  'requestion_num', 'rival_epcm', 'existingTime', 'rival_max_epcm', 'user_num',
                  'rival_pctr', 'rival_filter_rate', 'rival_q_epcm']

for col in train.columns:
    if ('_2day' in col or '_3day' in col):
        features_names.append(col)

# 标记部分特征为类别特征
cat_features = ['item_category', 'ad_size_id', 'ad_account_id', 'item_id', 'cost_mode', 'object']


train_x = train[features_names]
train_y = train.Y


valid_x = valid[features_names]
valid_y = valid.Y


'''
    lgb模型的目标函数
'''

def special_Objective(y_true, y_pred):

    y_true = pd.Series(y_true).apply(lambda x: max(x, 0.1)).values
    y_pred = np.array(y_pred)

    err = y_pred - y_true
    err_sign = np.sign(err)

    err_abs = np.abs(err)
    #150 88.85 120
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
param = {
    'random_state': 2019,
    'boosting_type': 'gbdt',
    'objective': special_Objective,
    'num_leaves': 1024,
    'max_depth': -1,
    'learning_rate': 0.05,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 600,
    'reg_alpha': 10,
}

#建立模型
lgb_model = lgb.LGBMRegressor(**param)
lgb_model = singleModel(lgb_model, proba=False, kfold=KFold(n_splits=5, random_state=0))

#模型训练
lgb_model.fit(train_x, train_y, metric=lambda x, y: Online_Metric(x, y)[1],
              valid_x=valid_x, valid_y=valid_y,
              categorical_feature=cat_features,
              eval_set_param_name='eval_set', eval_metric=Online_Metric, verbose=50, train_pred_dim=1)

# 读取lgb_fm以及nn的预测结果
test_pre_nn = pd.read_csv('./processedData/static_nn_test_predict_val.csv')['model_predict']
test_pre_lgb_fm = pd.read_csv('./processedData/lgb_fm_test_predict_val.csv')['model_predict']

# 读取线上测试数据
test = pd.read_csv('processedData/test.csv')
test.create_time = pd.to_datetime(test.create_time)

# 对三个预测结果进行加权
test['model_predict'] = 0.65*test_pre_lgb_fm + 0.35*(0.7*lgb_model.predict(test[train_x.columns]) + 0.3*test_pre_nn)
test['predict_val'] = test.progress_apply(lambda x: x.model_predict*1.0  if x.create_time.ceil('D') < pd.to_datetime('2019-04-25') else x.model_predict*1.0, axis=1)

# 对最后的结果进行四舍五入
test['predict_val'] = np.round(test.predict_val)

# 根据当前加权且四舍五入后的结果还有广告的出价来生成提交的答案
test_df_to_answer(test)
