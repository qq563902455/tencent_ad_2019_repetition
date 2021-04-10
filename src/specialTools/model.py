import numpy as np

from sklearn.model_selection import KFold
from lxyTools.singleModelUtils import singleModel
from specialTools.answer import Online_Metric
from lxyTools.pytorchTools import set_random_seed

import torch
import torch.nn as nn
import time



from tqdm import tqdm
import visdom


class classificationRegressor:
    def __init__(self, classifier, regressor, reg_ratio, kfold_cf, kfold_reg):

        self.classifier = singleModel(classifier, kfold_cf)
        self.regressor = singleModel(regressor, proba=False, kfold=kfold_reg)
        self.reg_ratio = reg_ratio

    def fit(self, X, Y, cf_metric, reg_metric,
            cf_features=None,
            reg_features=None,
            eval_set_param_name_cf=None,
            eval_set_param_name_reg=None,
            param_fit_dict_cf={},
            param_fit_dict_reg={}):


        if cf_features is None: cf_features = X.columns
        if reg_features is None: reg_features = X.columns

        self.cf_features = cf_features
        self.reg_features = reg_features

        self.classifier.fit(X[cf_features], Y>1, metric=cf_metric, train_pred_dim=2,
                            eval_set_param_name=eval_set_param_name_cf,
                            **param_fit_dict_cf)

        self.threshold = np.percentile(self.classifier.train_pred[:, 1], self.reg_ratio)

        reg_X = X[self.classifier.train_pred[:, 1]>self.threshold]
        reg_Y = Y[self.classifier.train_pred[:, 1]>self.threshold]
        self.reg_Y = reg_Y

        self.regressor.fit(reg_X[reg_features], reg_Y, metric=reg_metric, train_pred_dim=1,
                           eval_set_param_name=eval_set_param_name_reg,
                           **param_fit_dict_reg)


        train_pred = np.zeros(self.classifier.train_pred.shape[0])
        train_pred[self.classifier.train_pred[:, 1]>self.threshold] = self.regressor.train_pred[:, 0]

        print('score of cfRegressor:\t', reg_metric(Y, train_pred))

    def predict(self, X):
        out = np.zeros(X.shape[0])
        cf_proba = self.classifier.predict_proba(X[self.cf_features])[:, 1]
        reg_out = self.regressor.predict(X[cf_proba>self.threshold][self.reg_features])
        out[cf_proba>self.threshold] = reg_out
        return out, cf_proba



class nn_model_with_kfold:
    def __init__(self, modelFun, param):
        self.modelFun = modelFun
        self.param = param

    def fit(self, train_x, train_y, valid_x, valid_y, kfold, train_epochs, batch_size):

        seed_start = 10086
        seed_step = 500

        self.fittedModelslist = []

        custom_metric=lambda x,y : Online_Metric(x, y)[1]
        train_preds = np.zeros((len(train_x)))

        for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):

            x_train_fold = train_x[train_idx]
            y_train_fold = train_y[train_idx]
            x_val_fold = train_x[valid_idx]
            y_val_fold = train_y[valid_idx]

            model = self.modelFun(random_seed=seed_start+fold_num*seed_step, **self.param)
            model = model.cuda()
            model.batch_size = batch_size
            vis = visdom.Visdom(env=model.vis)

            set_random_seed(model.random_seed)

            x_train_fold_tensor = torch.tensor(x_train_fold, dtype=torch.float32).cuda()
            y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.float32).cuda()
            x_val_fold_tensor = torch.tensor(x_val_fold, dtype=torch.float32).cuda()
            y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.float32).cuda()

            loss_fn = model.loss_fn
            optimizer = model.optimizer
            scheduler = model.scheduler

            train_fold = torch.utils.data.TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
            valid_fold = torch.utils.data.TensorDataset(x_val_fold_tensor, y_val_fold_tensor)

            train_fold_loader = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, shuffle=True)
            valid_fold_loader = torch.utils.data.DataLoader(valid_fold, batch_size=batch_size, shuffle=False)

            for epoch in range(train_epochs):
                scheduler.step()

                start_time = time.time()
                model.train()
                avg_loss = 0.
                for x_batch, y_batch in tqdm(train_fold_loader, disable=True):

                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    avg_loss += loss.item() / len(train_fold_loader)
                avg_loss /= batch_size

                model.eval()
                valid_fold_preds = np.zeros((y_val_fold.shape[0]))

                avg_val_loss = 0.
                for i, (x_batch, y_batch) in enumerate(valid_fold_loader):
                    y_pred = model(x_batch)
                    y_pred = y_pred.detach()
                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_fold_loader)
                    valid_fold_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()

                avg_val_loss /= batch_size


                kfold_score = custom_metric(y_val_fold, valid_fold_preds)
                valid_score = custom_metric(valid_y, model.predict_proba(valid_x))

                elapsed_time = time.time() - start_time


                vis.line(X=torch.Tensor([[epoch, epoch]]),
                         Y=torch.Tensor([[avg_loss, avg_val_loss]]),
                         win='loss'+'_'+str(fold_num),
                         opts={'legend':['local_loss', 'valid_loss'],
                               'xlabel': 'epoch',
                               'title': 'train'+'_'+str(fold_num)},
                         update='append' if epoch > 0 else None)


                vis.line(X=torch.Tensor([[epoch, epoch]]),
                         Y=torch.Tensor([[kfold_score, valid_score]]),
                         win='score'+'_'+str(fold_num),
                         opts={'legend':['kfold score', 'valid score'],
                               'xlabel': 'epoch',
                               'title': 'valid'+'_'+str(fold_num)},
                         update='append' if epoch > 0 else None)




                print('Epoch {}/{} \t loss={:.4f} \t val_fold_loss={:.4f} \t fold_score={:.4f} \t valid_score={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, train_epochs, avg_loss, avg_val_loss, kfold_score, valid_score, elapsed_time))


            train_preds[valid_idx] = model.predict_proba(x_val_fold)
            self.fittedModelslist.append(model)
            print('-'*50)

        print('-'*20, 'finished', '-'*20)
        print('kfold score:\t', custom_metric(train_y, train_preds))
        print('valid score:\t', custom_metric(valid_y, self.predict(valid_x)))
        print('-'*50)

    def predict(self, test):
        answerlist = [model.predict_proba(test) for model in self.fittedModelslist]
        return np.array(answerlist).mean(axis=0)



class model_base:
    def predict(self, x):
        x_cuda = [
                torch.tensor(x[0], dtype=torch.long).cuda(),
                torch.tensor(x[1], dtype=torch.float32).cuda(),
            ]
        test = torch.utils.data.TensorDataset(*x_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

        batch_size = self.batch_size

        test_preds = np.zeros(len(x[0]))
        for i, (x_batch_1, x_batch_2) in enumerate(test_loader):
            x_batch = [x_batch_1, x_batch_2]

            y_pred = self(*x_batch)
            y_pred = y_pred.detach()
            test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()

        return test_preds



class dsnn_model_with_kfold:
    def __init__(self, modelFun, param):
        self.modelFun = modelFun
        self.param = param

    def fit(self, train_x, train_y, valid_x, valid_y, kfold, train_epochs, batch_size):

        seed_start = 10086
        seed_step = 500

        self.fittedModelslist = []

        custom_metric=lambda x,y : Online_Metric(x, y)[1]
        train_preds = np.zeros((len(train_x[0])))

        for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(train_x[0], train_y)):
            x_train_fold = [data[train_idx] for data in train_x]
            y_train_fold = train_y[train_idx]

            x_val_fold = [data[valid_idx] for data in train_x]
            y_val_fold = train_y[valid_idx]

            model = self.modelFun(random_seed=seed_start+fold_num*seed_step, **self.param)
            model = model.cuda()
            model.batch_size = batch_size

            set_random_seed(model.random_seed)

            x_train_fold_tensor = [
                torch.tensor(x_train_fold[0], dtype=torch.long).cuda(),
                torch.tensor(x_train_fold[1], dtype=torch.float32).cuda(),
            ]


            x_val_fold_tensor = [
                torch.tensor(x_val_fold[0], dtype=torch.long).cuda(),
                torch.tensor(x_val_fold[1], dtype=torch.float32).cuda(),
            ]

            y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.float32).cuda()
            y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.float32).cuda()


            loss_fn = model.loss_fn
            optimizer = model.optimizer
            scheduler = model.scheduler

            train_fold = torch.utils.data.TensorDataset(*x_train_fold_tensor, y_train_fold_tensor)
            valid_fold = torch.utils.data.TensorDataset(*x_val_fold_tensor, y_val_fold_tensor)

            train_fold_loader = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, shuffle=True)
            valid_fold_loader = torch.utils.data.DataLoader(valid_fold, batch_size=batch_size, shuffle=False)

            for epoch in range(train_epochs):
                scheduler.step()

                start_time = time.time()
                model.train()
                avg_loss = 0.
                for x_batch_1, x_batch_2, y_batch in tqdm(train_fold_loader, disable=True):

                    x_batch = [x_batch_1, x_batch_2]
                    y_pred = model(*x_batch)
                    loss = loss_fn(y_pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    avg_loss += loss.item() / len(train_fold_loader)
                avg_loss /= batch_size

                model.eval()
                valid_fold_preds = np.zeros((y_val_fold.shape[0]))

                avg_val_loss = 0.
                for i, (x_batch_1, x_batch_2, y_batch) in enumerate(valid_fold_loader):
                    x_batch = [x_batch_1, x_batch_2]
                    y_pred = model(*x_batch)
                    y_pred = y_pred.detach()
                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_fold_loader)
                    valid_fold_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()

                avg_val_loss /= batch_size


                kfold_score = custom_metric(y_val_fold, valid_fold_preds)
                valid_score = custom_metric(valid_y, model.predict(valid_x))

                elapsed_time = time.time() - start_time


                print('Epoch {}/{} \t loss={:.4f} \t val_fold_loss={:.4f} \t fold_score={:.4f} \t valid_score={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, train_epochs, avg_loss, avg_val_loss, kfold_score, valid_score, elapsed_time))


            train_preds[valid_idx] = model.predict(x_val_fold)
            self.fittedModelslist.append(model)
            print('-'*50)

        print('-'*20, 'finished', '-'*20)
        print('kfold score:\t', custom_metric(train_y, train_preds))
        print('valid score:\t', custom_metric(valid_y, self.predict(valid_x)))
        print('-'*50)

    def predict(self, test):
        answerlist = [model.predict(test) for model in self.fittedModelslist]
        return np.array(answerlist).mean(axis=0)


class multiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        nn.Module.__init__(self)

        if type(kernel_size)==int:
            padding = [int((kernel_size - 1)/2)]
            kernel_size = [kernel_size]
        else:
            padding = [int((val - 1)/2) for val in kernel_size]

        out_channels = int(out_channels/len(kernel_size))

        modellist = []
        for kernel_size_val, padding_size in zip(kernel_size, padding):
            modellist.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size_val, padding=padding_size))

        self.modellist = nn.ModuleList(modellist)

    def forward(self, x):

        x = torch.transpose(x, 1, 2).contiguous()

        outlist = [model(x) for model in self.modellist]
        out = torch.cat(outlist, dim=1)
        out = torch.transpose(out, 1, 2).contiguous()

        return out




class lgb_nn_with_kfold:
    def __init__(self, lgb_model_fun, lgb_param, nn_model_fun, nn_param):

        self.lgb_model_fun = lgb_model_fun
        self.nn_model_fun = nn_model_fun

        self.lgb_param = lgb_param
        self.nn_param = nn_param

    def fit(self, train_x, train_y, cat_features, valid_x, valid_y, kfold, train_epochs, batch_size):

        seed_start = 10086
        seed_step = 500

        self.fittedLgbModellist = []
        self.fittedNNModellist = []

        self.train_pred = np.zeros(train_x.shape[0])

        for fold_num, (kTrainIndex, kTestIndex) in enumerate(kfold.split(train_x, train_y)):

            kTrain_x = train_x.iloc[kTrainIndex]
            kTrain_y = train_y.iloc[kTrainIndex]

            kTest_x = train_x.iloc[kTestIndex]
            kTest_y = train_y.iloc[kTestIndex]

            self.lgb_param['random_state'] = seed_start + fold_num * seed_step
            lgb_model = self.lgb_model_fun(**self.lgb_param)

            lgb_model.fit(kTrain_x, kTrain_y,categorical_feature=cat_features, eval_metric=Online_Metric, eval_set=[(kTest_x, kTest_y), (valid_x, valid_y)], verbose=50)

            kTrain_x_leaves = lgb_model.predict(kTrain_x, pred_leaf=True)
            kTest_x_leaves = lgb_model.predict(kTest_x, pred_leaf=True)
            valid_x_leaves = lgb_model.predict(valid_x, pred_leaf=True)

            for i in range(self.lgb_param['n_estimators']):
                kTrain_x_leaves[:, i] += i*self.lgb_param['num_leaves']
                kTest_x_leaves[:, i] += i*self.lgb_param['num_leaves']
                valid_x_leaves[:, i] += i*self.lgb_param['num_leaves']

            self.nn_param['random_seed'] = seed_start + fold_num * seed_step
            set_random_seed(self.nn_param['random_seed'])
            nn_model = self.nn_model_fun(**self.nn_param).cuda()

            nn_model.fit(kTrain_x_leaves, kTrain_y.values, train_epochs, batch_size, kTest_x_leaves, kTest_y.values, custom_metric=lambda x, y: Online_Metric(x, y)[1], plot_fold=str(fold_num))

            valid_pre = nn_model.predict(valid_x_leaves)
            self.train_pred[kTestIndex] = nn_model.predict(kTest_x_leaves)

            print('valid socre:\t', Online_Metric(valid_y, valid_pre)[1])
            print('--'*50)
            self.fittedLgbModellist.append(lgb_model)
            self.fittedNNModellist.append(nn_model)

        print('train kfold score:\t', Online_Metric(train_y, self.train_pred))
        print('valid score:\t', Online_Metric(valid_y, self.predict(valid_x)))

    def predict(self, x):
        out_list = []
        for lgb_model, nn_model in zip(self.fittedLgbModellist, self.fittedNNModellist):
            x_leaves = lgb_model.predict(x, pred_leaf=True)

            for i in range(self.lgb_param['n_estimators']):
                x_leaves[:, i] += i*self.lgb_param['num_leaves']

            out = nn_model.predict(x_leaves)
            out_list.append(out)

        return np.array(out_list).mean(axis=0)


class multiLgb:
    def __init__(self, lgb_model_fun, lgb_param_list, features_list, random_state, **kwargs):
        self.lgb_model_fun = lgb_model_fun
        self.lgb_param_list = lgb_param_list
        self.features_list = features_list

        for param_dict in self.lgb_param_list:
            param_dict['random_state'] = random_state

    def fit(self, train_x, train_y, categorical_feature, eval_metric, eval_set, verbose):

        self.fittedModelslist = []
        for param_dict, features in zip(self.lgb_param_list, self.features_list):
            model = self.lgb_model_fun(**param_dict)

            cat_features = [fe for fe in features if fe in categorical_feature]
            if len(cat_features) == 0: cat_features='auto'

            eval_set_single = [(x[features], y) for (x, y) in eval_set]

            model.fit(train_x[features], train_y, categorical_feature=cat_features, eval_metric=eval_metric, eval_set=eval_set_single, verbose=verbose)
            self.fittedModelslist.append(model)

    def predict(self, x, pred_leaf=True):
        assert pred_leaf==True, '这个模型单纯只是为了配合lgb_nn_with_kfold， 所以部分参数不能随意变动'

        relist = []
        for model,features in zip(self.fittedModelslist, self.features_list):
            re = model.predict(x[features], pred_leaf=True)
            relist.append(re)

        re = np.hstack(relist)
        return re
