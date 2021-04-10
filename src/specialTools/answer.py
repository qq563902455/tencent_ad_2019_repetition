import pandas as pd
import numpy as np
import math

def Online_Metric(y_true, y_pred):

    if type(y_pred) == float or type(y_pred) == int: y_pred = np.ones(y_true.shape[0])*y_pred
    if len(y_pred.shape)!=1: y_pred = y_pred.reshape(-1)

    y_pred = pd.Series(y_pred).apply(lambda x: max(x, 1)).values
    y_true = pd.Series(y_true).apply(lambda x: max(x, 1)).values

    err = np.abs(y_true - y_pred)
    base = (y_true + y_pred)/2

    score = np.mean(err/base)

    score = 40*(1-(score/2)) + 50
    assert score < 100,  print(y_true, y_pred)
    return ('online score', score, True)





def test_df_to_answer(test):
    test['bid_rank'] = test.groupby('ad_id').agg({'bid': 'rank'})
    num_bid = test.groupby('ad_id')['ad_id'].count()

    ad_id_predict_val_list = []
    for num_bid_val in num_bid.unique():
        print(num_bid_val, math.ceil(num_bid_val/2))
        ad_id_predict_val_list.append(test.loc[test.ad_id.isin(num_bid[num_bid==num_bid_val].index)&(test['bid_rank']==math.ceil(num_bid_val/2)), ['ad_id', 'predict_val']])

    ad_id_predict_val = pd.concat(ad_id_predict_val_list)

    test = pd.merge(test.drop(['predict_val'], axis=1), ad_id_predict_val, on='ad_id', how='left')
    test.predict_val = test.predict_val.apply(lambda x: max(x, 1))

    for num_bid_val in num_bid.unique():
        test.loc[test.ad_id.isin(num_bid[num_bid==num_bid_val].index), 'bid_rank'] = test[test.ad_id.isin(num_bid[num_bid==num_bid_val].index)].bid_rank - math.ceil(num_bid_val/2)

    test.predict_val = test.predict_val + test.bid_rank*0.0001


    test[['test_id', 'predict_val']].to_csv('./submission.csv', index=False, header=False)


def test_df_to_answer_with_bid(test):
    test['bid_rank'] = test.groupby('ad_id').agg({'bid': 'rank'})
    # num_bid = test.groupby('ad_id')['ad_id'].count()
    #
    # ad_id_predict_val_list = []
    # for num_bid_val in num_bid.unique():
    #     print(num_bid_val, math.ceil(num_bid_val/2))
    #     ad_id_predict_val_list.append(test.loc[test.ad_id.isin(num_bid[num_bid==num_bid_val].index)&(test['bid_rank']==math.ceil(num_bid_val/2)), ['ad_id', 'predict_val']])
    #
    # ad_id_predict_val = pd.concat(ad_id_predict_val_list)
    #
    # test = pd.merge(test.drop(['predict_val'], axis=1), ad_id_predict_val, on='ad_id', how='left')
    # test.predict_val = test.predict_val.apply(lambda x: max(x, 1))
    #
    # for num_bid_val in num_bid.unique():
    #     test.loc[test.ad_id.isin(num_bid[num_bid==num_bid_val].index), 'bid_rank'] = test[test.ad_id.isin(num_bid[num_bid==num_bid_val].index)].bid_rank - math.ceil(num_bid_val/2)

    test.predict_val = test.predict_val + test.bid_rank*0.0001


    test[['test_id', 'predict_val']].to_csv('./submission.csv', index=False, header=False)
