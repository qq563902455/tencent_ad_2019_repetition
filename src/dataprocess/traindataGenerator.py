import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('error')

from tqdm import tqdm

tqdm.pandas('bar')


'''
    读取数据
'''

rawdata_path = './rawdata/'
processed_path = './processedData/'

raw_train = pd.read_csv(processed_path+'trainset_cleaned.csv')
raw_train.days = pd.to_datetime(raw_train.days)
raw_train.create_time = pd.to_datetime(raw_train.create_time)


raw_test = pd.read_csv(processed_path+'testset_cleaned_b.csv')
raw_test.create_time = pd.to_datetime(raw_test.create_time)
raw_test['days'] = pd.to_datetime('2019-04-25')


exposure_df = pd.read_csv(processed_path+'exposure_df_cleaned.csv')
exposure_df.ad_exposure_time = pd.to_datetime(exposure_df.ad_exposure_time)
exposure_df['days'] = exposure_df.ad_exposure_time.dt.ceil('D')
exposure_df = exposure_df.rename({'exposed_ad_id': 'ad_id'}, axis=1)


names = [
    'ad_id', 'create_time', 'ad_account_id', 'item_id', 'item_category',
    'industry_id', 'ad_size_id'
]
dtype = {
    'ad_id': np.int32,
    'ad_account_id': np.int32,
    'item_id': np.int32,
    'item_category': np.int32,
    'industry_id': np.int32,
    'ad_size_id': np.int32,
}

ad_static_df = pd.read_csv(rawdata_path+'total_data/map_ad_static.out', sep='\t', names=names, dtype=dtype, keep_default_na=False)
ad_static_df.loc[ad_static_df.create_time == -1, 'create_time'] = np.nan
ad_static_df.create_time = pd.to_datetime(ad_static_df.create_time+8*3600, unit='s')

print('shape of ad_static_df:\t', ad_static_df.shape)
exposure_df = exposure_df.merge(ad_static_df[['ad_id', 'ad_account_id', 'item_id', 'ad_size_id', 'industry_id']], on='ad_id', how='left')



'''
    划分训练集和验证集范围
'''

time_valid = pd.to_datetime('2019-04-22')

train_df = raw_train[raw_train.days<time_valid]
valid_df = raw_train[raw_train.days>=time_valid]





def df_get_features(df, isTest=False):
    df = df.copy(True)

    # 获取每一个广告从创建到要预测的时间，总共有多长时间,即为每一个广告的存在时间
    df['existingTime'] = (df.days - df.create_time)/np.timedelta64(1, 's')

    return df



def df_get_features_from_exposure_df(df, exposure_df, isTest=False):
    df = df.copy(True)

    if isTest:
        history_period_list = [2, 3]
    else:
        history_period_list = [1, 2, 3]

    max_period = max(history_period_list)
    if not isTest:
        df = df[df.days>(pd.to_datetime('2019-04-10')+(pd.Timedelta('1 days')*max_period))]

    # 统计每一个广告在前天以及大前天的曝光情况
    for col in ['ad_id', 'ad_account_id', 'item_id', 'ad_size_id']:
        for i in history_period_list:
            new_col_name = col+'_exposure_'+str(i)+'day'

            temp_ex_df = exposure_df

            if col == 'ad_id':
                # ad_id 就是 统计前天该广告曝光量以及前天该广告曝光量
                history_df = temp_ex_df.groupby(by=[col, 'days'], as_index=False).agg({'user_id': 'count'}).rename({'user_id': new_col_name}, axis=1)
            else:
                # 非ad_id时，例如ad_account_id就是统计一个广告对应账户在前天以及大前天旗下广告的曝光量的中位数
                history_df = temp_ex_df.groupby(by=[col, 'ad_id', 'days'], as_index=False).agg({'user_id': 'count'}).rename({'user_id': new_col_name}, axis=1)
                history_df = history_df.groupby(by=[col, 'days'], as_index=False).agg({new_col_name: 'median'})

            history_df.days = history_df.days + i*pd.Timedelta('1 days')

            df = df.merge(history_df, on=[col, 'days'], how='left')
            df[new_col_name] = df[new_col_name].fillna(0)

    '''
        这一部分是多个特征的交互特征
        例如 ad_id filter_rate 就是统计该广告在前天以及大前天的被屏蔽概率
            ad_account_id filter_rate 就是统计该广告对应账户在前天以及大前天旗下广告被屏蔽概率的中位数
    '''
    for col in ['ad_id', 'ad_account_id', 'item_id']:
        for sub_col in ['filter_rate', 'rival_epcm', 'rival_max_epcm',
                        'user_num','rival_pctr', 'rival_filter_rate', 'rival_q_epcm']:
            for i in history_period_list:
                new_col_name = col+'_'+sub_col+'_'+str(i)+'day'

                if col == 'ad_id':
                    history_df = raw_train[['ad_id', 'days', sub_col]].copy(True)
                    history_df.days = history_df.days + i*pd.Timedelta('1 days')
                else:
                    history_df = raw_train.groupby([col, 'days'], as_index=False).agg({sub_col: 'median'})
                    history_df.days = history_df.days + i*pd.Timedelta('1 days')

                history_df = history_df.rename({sub_col: new_col_name}, axis=1)

                df = df.merge(history_df, on=[col, 'days'], how='left')
                df[new_col_name] = df[new_col_name].fillna(-1)

    # 这里也是交互特征，但是统计的是标准差而不是中位数
    for col in ['ad_account_id']:
        for sub_col in ['filter_rate', 'rival_epcm', 'rival_max_epcm',
                        'user_num','rival_pctr', 'rival_filter_rate']:
            for i in history_period_list:
                new_col_name = col+'_'+sub_col+'_std_'+str(i)+'day'

                history_df = raw_train.groupby([col, 'days'], as_index=False).agg({sub_col: 'std'})
                history_df.days = history_df.days + i*pd.Timedelta('1 days')

                history_df = history_df.rename({sub_col: new_col_name}, axis=1)
                # print(history_df.columns)
                df = df.merge(history_df, on=[col, 'days'], how='left')
                df[new_col_name] = df[new_col_name].fillna(-1)

    return df

# 调用上面的函数进行特征提取
train_df = df_get_features(train_df)
valid_df = df_get_features(valid_df)
raw_test = df_get_features(raw_test, isTest=True)

train_df = df_get_features_from_exposure_df(train_df, exposure_df)
valid_df = df_get_features_from_exposure_df(valid_df, exposure_df)

raw_test = df_get_features_from_exposure_df(raw_test, exposure_df, isTest=True)

# 输出文件
train_df.to_csv(processed_path+'train.csv', index=False)
valid_df.to_csv(processed_path+'valid.csv', index=False)
raw_test.to_csv(processed_path+'test.csv', index=False)
