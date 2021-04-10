import pandas as pd
import numpy as np
import math
import time

import sys
sys.path.append('./src')


from tqdm import tqdm

import torch
import torch.nn as nn

from lxyTools.singleModelUtils import singleModel
from lxyTools.adversarialValidation import adversarialValidation
from lxyTools.pytorchTools import set_random_seed
from lxyTools.pytorchTools import myBaseModule
from lxyTools.pytorchTools import Attention
from lxyTools.pytorchTools import BiInteraction
from lxyTools.pytorchTools import CrossNet
from lxyTools.pytorchTools import FM
from lxyTools.pytorchTools import CompressedInteractionNet

from specialTools.model import classificationRegressor
from specialTools.model import nn_model_with_kfold
from specialTools.model import model_base
from specialTools.model import dsnn_model_with_kfold
from specialTools.model import multiConv

from specialTools.answer import Online_Metric
from specialTools.answer import test_df_to_answer

from specialTools.preprocessor import id_index_converter
from specialTools.preprocessor import continous_col_converter
from specialTools.preprocessor import id_count_convert
from specialTools.preprocessor import continous_col_discretizer


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from tqdm import tqdm
tqdm.pandas('bar')

# 读取数据
train = pd.read_csv('./processedData/train.csv')
train.create_time = pd.to_datetime(train.create_time)
train.days = pd.to_datetime(train.days)

# 对数据中曝光量为0的样本进行降采样
train_select_0 = train[train.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
train_select_1 = train[train.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
train_select = train_select_0 + train_select_1
train = train.loc[train_select]
train = train.sample(frac=1.0, random_state=111)

# 读取数据
valid = pd.read_csv('./processedData/valid.csv')
valid.create_time = pd.to_datetime(valid.create_time)
valid.days = pd.to_datetime(valid.days)

# 对数据中曝光量为0的样本进行降采样
valid_select_0 = valid[valid.Y==0].sample(frac=0.4, random_state=2014).index.tolist()
valid_select_1 = valid[valid.Y>0].sample(frac=1.0, random_state=2014).index.tolist()
valid_select = valid_select_0 + valid_select_1
valid = valid.loc[valid_select]
valid = valid.sample(frac=1.0, random_state=111)

# 读取线上测试数据
test = pd.read_csv('processedData/test.csv')
test.create_time = pd.to_datetime(test.create_time)

# 将这两个取值非常多的特征转换成频数
id_cols=['item_id', 'ad_account_id']
id_count = id_count_convert(dataset_list=[train, valid, test], id_cols=id_cols)
train[id_cols] = id_count.transform(train)
valid[id_cols] = id_count.transform(valid)
test[id_cols] = id_count.transform(test)

# 离散特征embedding前预处理数据
id_converter = id_index_converter(dataset_list=[train, valid, test],
                                  id_col=['item_category', 'ad_size_id', 'cost_mode', 'object'])


# 连续特征选择
features_names = ['requestion_num', 'rival_epcm', 'existingTime', 'rival_max_epcm',
                  'user_num', 'rival_pctr', 'rival_filter_rate', 'rival_q_epcm', 'item_id', 'ad_account_id']


for col in train.columns:
    if ('_2day' in col or '_3day' in col):
        features_names.append(col)

# 连续特征预处理
con_converter = continous_col_converter(dataset_list=[train, valid, test], cols=features_names, percentile=95)

# 统计所有离散特征，需要的embedding_size
embedding_size_list = []
for key in id_converter.cat_list_dict.keys():
    embedding_size_list.append(len(id_converter.cat_list_dict[key]))

# 数据处理
extract_features = lambda x: [id_converter.transform(x), con_converter.transform(x)]

train_x = extract_features(train)
valid_x = extract_features(valid)
test_x = extract_features(test)

train_y = train.Y
valid_y = valid.Y

# 计算标签的根号结果的90%分位数,这个用于网络输出的缩放
out_ratio = np.percentile(np.sqrt(train_y), 90)


# 离散特征进行embedding的模型
class idModel(nn.Module):
    def __init__(self, embedding_sizelist, embedding_dimlist):
        nn.Module.__init__(self)
        self.embeddinglist = nn.ModuleList(
            [nn.Embedding(size, dim).cuda() for size, dim in zip(embedding_sizelist, embedding_dimlist)]
        )

    def forward(self, x):
        embeddinglist = []
        for i, embedding in enumerate(self.embeddinglist):
            embeddinglist.append(embedding(x[:, i]))

        embedding = torch.cat(embeddinglist, dim=1)
        return embedding



# 网络结构
class compoundModel(nn.Module, model_base):
    def __init__(self, random_seed,
                 embedding_sizelist, embedding_dimlist,
                 c_input_dim,
                 learning_rate, out_ratio):
        nn.Module.__init__(self)

        self.random_seed = random_seed
        set_random_seed(random_seed)

        self.out_ratio = out_ratio

        self.id_model = idModel(embedding_sizelist, embedding_dimlist)
        id_out_size = sum(embedding_dimlist)

        self.dnn_feature_extract = nn.Sequential(
                    nn.Linear(c_input_dim + id_out_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 600),
                    nn.Dropout(0.05),
                    nn.ReLU(),
        )


        self.dnn_predict_val = nn.Sequential(
                    nn.Linear(600+c_input_dim + id_out_size, 1, bias=False),
        )

        self.out_bias = torch.tensor(1, dtype=torch.float32)
        self.out_bias = nn.Parameter(self.out_bias)

        def loss_fn(x, y):

            x = torch.nn.functional.leaky_relu(x -1) + 1
            y = torch.nn.functional.relu(y -1) + 1

            re = torch.abs(x - y)/(torch.abs(x + y))

            return torch.sum(re)

        self.loss_fn = loss_fn
        # self.loss_fn = lambda x, y: torch.sum(torch.abs(torch.abs(x -1) - torch.abs(y - 1))/(torch.abs(x - 1) + torch.abs(y - 1) + 2))
        # self.loss_fn = torch.nn.MSELoss(reduction='sum')
        # self.loss_fn = lambda x, y: torch.sum((x - y)**2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))

        self.vis = None

    def forward(self, x_id, x_static):

        x_id = self.id_model(x_id)

        x = torch.cat([x_id, x_static], dim=1)

        dnn_out = self.dnn_feature_extract(x)
        dnn_out = torch.cat([dnn_out, x], dim=1)

        out = self.dnn_predict_val(dnn_out) + self.out_bias
        return (out.reshape(-1) * self.out_ratio)**2


# kfold
kfold = KFold(n_splits=5, shuffle=True, random_state=10)

# 模型参数
param = {
    'embedding_sizelist': embedding_size_list,
    'embedding_dimlist':[2, 2, 1, 2],

    'c_input_dim': train_x[1].shape[1],

    'learning_rate': 0.001,
    'out_ratio': out_ratio
}

# 模型创建以及训练
model = dsnn_model_with_kfold(compoundModel, param)
model.fit([data for data in train_x], train_y.values,
          [data for data in valid_x], valid_y.values, kfold, 80, 2048)


# 读取线上数据
test = pd.read_csv('processedData/test.csv')
test.create_time = pd.to_datetime(test.create_time)

# 预测线上数据
test['model_predict'] = model.predict(test_x)
test[['test_id', 'model_predict']].to_csv('./processedData/static_nn_test_predict_val.csv', index=False)
