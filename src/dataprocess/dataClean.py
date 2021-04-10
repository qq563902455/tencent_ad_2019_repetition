# import modin.pandas as pd
import pandas as pd
import numpy as np
import time
import gc
import sys
import os
from multiprocessing import Pool

from tqdm import tqdm
tqdm.pandas(desc="my bar!")

# 路径设置,读取数据路径，以及输出数据路径
rawdata_path = './rawdata/'
output_path = './processedData/'

names = [
    'ad_user_time_id', 'ad_exposure_time', 'user_id', 'ad_position_id', 'ad_info',
]
dtype = {
    'ad_user_time_id': np.int32,
    'ad_position_id': np.int16,
    'user_id':  np.int32,
    'ad_info': str,
}

exposure_df_list = []
ad_id_df_list = []

for i in range(10, 23):
    # 读取曝光请求数据
    exposure_df = pd.read_csv(rawdata_path+'total_data/track_log/track_log_201904'+str(i)+'.out', sep='\t', names=names, dtype=dtype)
    exposure_df.ad_exposure_time = pd.to_datetime(exposure_df.ad_exposure_time+8*3600, unit='s')
    exposure_df['days'] = exposure_df.ad_exposure_time.max().ceil('D')

    # 统计每一个广告在当天数据里面出现的次数，即为当天请求次数
    all_id_dict = {}
    def get_all_id(x):
        global all_id_dict
        for val in x:
            if all_id_dict.get(val) is None:
                all_id_dict[val] = 1
            else:
                all_id_dict[val] = all_id_dict[val] + 1

    exposure_df.ad_info.progress_apply(lambda x: [vallist.split(',')[0] for vallist in x.split(';')]).apply(get_all_id)

    all_id_list = list(all_id_dict.keys())
    all_id_requestion_num = list(all_id_dict.values())
    ad_id_df = pd.DataFrame({'ad_id': all_id_list, 'requestion_num': all_id_requestion_num})
    ad_id_df['days'] = exposure_df.ad_exposure_time.max().ceil('D')

    # 统计每一个广告的所有曝光请求对应的不同用户数目
    user_id_ad_id_df = exposure_df.groupby(by=['user_id'], as_index=False).agg({'ad_info': lambda x: list(set(x.apply(lambda x: [vallist.split(',')[0] for vallist in x.split(';')]).sum()))})
    user_num_dict = {}
    def get_user_num(x):
        for ad_id in x:
            if user_num_dict.get(ad_id) is None:
                user_num_dict[ad_id] = 1
            else:
                user_num_dict[ad_id] += 1

    user_id_ad_id_df.ad_info.apply(get_user_num)

    ad_id_user_num_df = pd.DataFrame({'ad_id': list(user_num_dict.keys()), 'user_num': list(user_num_dict.values())})

    ad_id_df = ad_id_df.merge(ad_id_user_num_df, how='left', on='ad_id')


    ad_id_attribute_dict = {}
    def get_attribute_sum(x, attribute_index=-3):
        global ad_id_attribute_dict
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id = vallist[0]
            attribute_val = float(vallist[attribute_index])

            if ad_id_attribute_dict.get(ad_id) is None:
                ad_id_attribute_dict[ad_id] = attribute_val
            else:
                ad_id_attribute_dict[ad_id] += attribute_val

    # 统计每一个广告当天的平均被屏蔽概率
    ad_id_attribute_dict = {}
    exposure_df.ad_info.apply(lambda x: get_attribute_sum(x, -2))
    all_id_attribute_sum_df = pd.DataFrame({'ad_id': list(ad_id_attribute_dict.keys()), 'filter_rate': list(ad_id_attribute_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_attribute_sum_df, on='ad_id', how='left')
    ad_id_df['filter_rate'] = ad_id_df['filter_rate']/ad_id_df['requestion_num']

    # 统计每一个广告当天的平均总分数
    ad_id_attribute_dict = {}
    exposure_df.ad_info.apply(lambda x: get_attribute_sum(x, -3))
    all_id_attribute_sum_df = pd.DataFrame({'ad_id': list(ad_id_attribute_dict.keys()), 'total_ecpm': list(ad_id_attribute_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_attribute_sum_df, on='ad_id', how='left')
    ad_id_df['total_ecpm'] = ad_id_df['total_ecpm']/ad_id_df['requestion_num']

    # 统计每一个广告当天的平均质量得分
    ad_id_attribute_dict = {}
    exposure_df.ad_info.apply(lambda x: get_attribute_sum(x, -4))
    all_id_attribute_sum_df = pd.DataFrame({'ad_id': list(ad_id_attribute_dict.keys()), 'q_ecpm': list(ad_id_attribute_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_attribute_sum_df, on='ad_id', how='left')
    ad_id_df['q_ecpm'] = ad_id_df['q_ecpm']/ad_id_df['requestion_num']

    # 统计每一个广告当天的平均点击率
    ad_id_attribute_dict = {}
    exposure_df.ad_info.apply(lambda x: get_attribute_sum(x, -5))
    all_id_attribute_sum_df = pd.DataFrame({'ad_id': list(ad_id_attribute_dict.keys()), 'pctr': list(ad_id_attribute_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_attribute_sum_df, on='ad_id', how='left')
    ad_id_df['pctr'] = ad_id_df['pctr']/ad_id_df['requestion_num']


    rival_avg_epcm_dict = {}
    def get_rival_epcm_avg(x):
        global rival_avg_epcm_dict

        attribute_val = 0
        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id_list.append(vallist[0])
            attribute_val += float(vallist[-3])

        attribute_val = attribute_val/len(ad_id_list)

        for ad_id in ad_id_list:
            if rival_avg_epcm_dict.get(ad_id) is None:
                rival_avg_epcm_dict[ad_id] = attribute_val
            else:
                rival_avg_epcm_dict[ad_id] += attribute_val

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告的总得分平均值的平均值
    rival_avg_epcm_dict = {}
    exposure_df.ad_info.apply(get_rival_epcm_avg)
    all_id_rival_epcm_avg_sum_df = pd.DataFrame({'ad_id': list(rival_avg_epcm_dict.keys()), 'rival_epcm': list(rival_avg_epcm_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_epcm_avg_sum_df, on='ad_id', how='left')
    ad_id_df['rival_epcm'] = ad_id_df['rival_epcm']/ad_id_df['requestion_num']


    rival_avg_pctr_dict = {}
    def get_rival_pctr_avg(x):
        global rival_avg_pctr_dict

        attribute_val = 0
        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id_list.append(vallist[0])
            attribute_val += float(vallist[-5])

        attribute_val = attribute_val/len(ad_id_list)

        for ad_id in ad_id_list:
            if rival_avg_pctr_dict.get(ad_id) is None:
                rival_avg_pctr_dict[ad_id] = attribute_val
            else:
                rival_avg_pctr_dict[ad_id] += attribute_val

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告的点击率平均值的平均值
    rival_avg_pctr_dict = {}
    exposure_df.ad_info.apply(get_rival_pctr_avg)
    all_id_rival_pctr_avg_sum_df = pd.DataFrame({'ad_id': list(rival_avg_pctr_dict.keys()), 'rival_pctr': list(rival_avg_pctr_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_pctr_avg_sum_df, on='ad_id', how='left')
    ad_id_df['rival_pctr'] = ad_id_df['rival_pctr']/ad_id_df['requestion_num']


    rival_avg_q_epcm_dict = {}
    def get_rival_q_epcm_avg(x):
        global rival_avg_q_epcm_dict

        attribute_val = 0
        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id_list.append(vallist[0])
            attribute_val += float(vallist[-4])

        attribute_val = attribute_val/len(ad_id_list)

        for ad_id in ad_id_list:
            if rival_avg_q_epcm_dict.get(ad_id) is None:
                rival_avg_q_epcm_dict[ad_id] = attribute_val
            else:
                rival_avg_q_epcm_dict[ad_id] += attribute_val

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告的质量得分平均值的平均值
    rival_avg_q_epcm_dict = {}
    exposure_df.ad_info.apply(get_rival_q_epcm_avg)
    all_id_rival_q_epcm_avg_sum_df = pd.DataFrame({'ad_id': list(rival_avg_q_epcm_dict.keys()), 'rival_q_epcm': list(rival_avg_q_epcm_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_q_epcm_avg_sum_df, on='ad_id', how='left')
    ad_id_df['rival_q_epcm'] = ad_id_df['rival_q_epcm']/ad_id_df['requestion_num']



    rival_avg_filter_rate_dict = {}
    def get_rival_filter_rate_avg(x):
        global rival_avg_filter_rate_dict

        attribute_val = 0
        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id_list.append(vallist[0])
            attribute_val += float(vallist[-2])

        attribute_val = attribute_val/len(ad_id_list)

        for ad_id in ad_id_list:
            if rival_avg_filter_rate_dict.get(ad_id) is None:
                rival_avg_filter_rate_dict[ad_id] = attribute_val
            else:
                rival_avg_filter_rate_dict[ad_id] += attribute_val

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告的被屏蔽率平均值的平均值
    rival_avg_filter_rate_dict = {}
    exposure_df.ad_info.apply(get_rival_filter_rate_avg)
    all_id_rival_filter_rate_avg_sum_df = pd.DataFrame({'ad_id': list(rival_avg_filter_rate_dict.keys()), 'rival_filter_rate': list(rival_avg_filter_rate_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_filter_rate_avg_sum_df, on='ad_id', how='left')
    ad_id_df['rival_filter_rate'] = ad_id_df['rival_filter_rate']/ad_id_df['requestion_num']


    rival_max_epcm_dict = {}
    def get_rival_epcm_max(x):
        global rival_max_epcm_dict

        attribute_val = 0
        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')

            ad_id_list.append(vallist[0])
            attribute_val = max(float(vallist[-3]), attribute_val)


        for ad_id in ad_id_list:
            if rival_max_epcm_dict.get(ad_id) is None:
                rival_max_epcm_dict[ad_id] = attribute_val
            else:
                rival_max_epcm_dict[ad_id] += attribute_val

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告的最大总得分的平均值
    rival_max_epcm_dict = {}
    exposure_df.ad_info.apply(get_rival_epcm_max)
    all_id_rival_epcm_max_sum_df = pd.DataFrame({'ad_id': list(rival_max_epcm_dict.keys()), 'rival_max_epcm': list(rival_max_epcm_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_epcm_max_sum_df, on='ad_id', how='left')
    ad_id_df['rival_max_epcm'] = ad_id_df['rival_max_epcm']/ad_id_df['requestion_num']


    rival_num_dict = {}
    def get_rival_num(x):
        global rival_num_dict

        ad_id_list = []
        for vallist in x.split(';'):
            vallist = vallist.split(',')
            ad_id_list.append(vallist[0])


        for ad_id in ad_id_list:
            if rival_num_dict.get(ad_id) is None:
                rival_num_dict[ad_id] = len(ad_id_list)
            else:
                rival_num_dict[ad_id] += len(ad_id_list)

    # 统计每一个广告当天参与的所有的曝光竞争队列中广告数目的平均值
    rival_num_dict = {}
    exposure_df.ad_info.apply(get_rival_num)
    all_id_rival_num_sum_df = pd.DataFrame({'ad_id': list(rival_num_dict.keys()), 'rival_num': list(rival_num_dict.values())})

    ad_id_df = ad_id_df.merge(all_id_rival_num_sum_df, on='ad_id', how='left')
    ad_id_df['rival_num'] = ad_id_df['rival_num']/ad_id_df['requestion_num']

    ad_id_df_list.append(ad_id_df)

    # 统计出每一天曝光请求胜出的广告id这些将用于特征提取和标签生成
    exposure_df['exposed_ad_id'] = exposure_df.ad_info.progress_apply(lambda x: int([vallist.split(',')[0] for vallist in x.split(';') if vallist[-1] == '1'][0]))
    exposure_df = exposure_df.drop(['ad_info'], axis=1)

    exposure_df_list.append(exposure_df)
    gc.collect()
    print(exposure_df.ad_exposure_time.max().ceil('D'))

exposure_df = pd.concat(exposure_df_list)
exposure_df.to_csv('./processedData/exposure_df_cleaned.csv', index=False)

# 生成训练数据
ad_id_df = pd.concat(ad_id_df_list)
ad_id_df.ad_id = ad_id_df.ad_id.astype(int)
label_info = exposure_df.groupby(by=['exposed_ad_id', 'days'], as_index=False).agg({'user_id': 'count'}).rename({'user_id': 'Y', 'exposed_ad_id': 'ad_id'}, axis=1)

label_info = ad_id_df.merge(label_info, on=['ad_id', 'days'], how='left').fillna(0)

# 读取广告静态数据，并合并到训练数据上
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

label_info = pd.merge(label_info, ad_static_df[['ad_id', 'create_time', 'ad_account_id', 'item_id', 'item_category', 'industry_id', 'ad_size_id']], how='left', on='ad_id')

# 读取广告操作数据
names = [
    'ad_id', 'time', 'operation', 'object', 'cost_mode', 'bid'
]
dtype = {

}
ad_operation = pd.read_csv(rawdata_path+'total_data/final_map_bid_opt.out', sep='\t', names=names, dtype=dtype)
ad_operation.time = pd.to_datetime(ad_operation.time, format='%Y%m%d%H%M%S', errors='coerce')



# 该函数用于某一个广告，当天的广告设置
def getSetting(x):

    ad_operation_slice = ad_operation[ad_operation.ad_id == x.ad_id]
    ad_operation_slice = ad_operation_slice[ad_operation_slice.time < x.days]
    ad_operation_slice = ad_operation_slice.sort_values('time')
    if ad_operation_slice.shape[0]>0:
        return [ad_operation_slice.object.iloc[-1], ad_operation_slice.cost_mode.iloc[-1], ad_operation_slice.bid.iloc[-1]]
    else:
        return [np.nan, np.nan, np.nan]

# 多进程执行上面提到的这个函数，加速设置提取
result = 0
with Pool(processes=6) as pool:
    with tqdm([label_info.iloc[i] for i in range(len(label_info))]) as bar:
        result = pool.map(getSetting, bar, 2000)

# 合并设置信息和训练数据
label_info['object'] = [val[0] for val in result]
label_info['cost_mode'] = [val[1] for val in result]
label_info['bid'] = [val[2] for val in result]

label_info = label_info.dropna()
label_info.to_csv(output_path+'trainset_cleaned.csv', index=False)

# 测试集数据清洗和特征提取
for (test_index, sample_path, select_path, exposure_path) in [
    ['b', rawdata_path+'BTest/Btest_sample_bid.out', rawdata_path+'BTest/Btest_select_request_20190424.out', rawdata_path+'BTest/BTest_tracklog_20190424.txt'],
]:

    # 读取数据
    names = [
        'test_id', 'ad_id', 'object', 'cost_mode', 'bid',
    ]
    test_df = pd.read_csv(sample_path, sep='\t', names=names)
    test_df = test_df.merge(ad_static_df, on='ad_id', how='left')



    names = [
        'ad_id',
        'requestion_list'
    ]
    test_ad_id_select = pd.read_csv(select_path, sep='\t', names=names)
    # 统计每一个广告在所有曝光请求里面出现的次数
    test_ad_id_select['requestion_num'] = test_ad_id_select.requestion_list.apply(lambda x: x.count('|')+1)



    names = [
        'ad_user_time_id', 'ad_exposure_time', 'user_id', 'ad_position_id', 'ad_info',
    ]
    dtype = {
        'ad_user_time_id': np.int32,
        'ad_position_id': np.int16,
        'user_id':  np.int32,
        'ad_info': str,
    }
    exposure_df = pd.read_csv(exposure_path, sep='\t', names=names, dtype=dtype)
    exposure_df.ad_exposure_time = pd.to_datetime(exposure_df.ad_exposure_time+8*3600, unit='s')



    # 统计每一个广告的不同用户数目
    def get_user_num(x):
        requestion_list = [int(vallist.split(',')[0]) for vallist in x.split('|')]

        user_id_list = exposure_df[exposure_df.ad_user_time_id.isin(requestion_list)].user_id.tolist()

        return len(set(user_id_list))

    test_ad_id_select['user_num'] = test_ad_id_select.requestion_list.progress_apply(get_user_num)



    epcm_mean_df = exposure_df.ad_info.apply(lambda x: np.mean([float(vallist.split(',')[-2]) for vallist in x.split(';')]))
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)

    without_dict = {}
    def get_rival_epcm(x):

        relist = []

        for vallist in x.split('|'):
            requestion_id = vallist.split(',')[0]
            if requestion_id in epcm_mean_df.index:
                relist.append(epcm_mean_df.loc[requestion_id])
            else:
                without_dict[requestion_id] = 0

        return np.mean(relist)
    #  每个广告对应所有请求曝光队列中广告总体得分的平均值的平均值
    test_ad_id_select['rival_epcm'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)


    #  每个广告对应所有请求曝光队列中广告总体得分的最大值的平均值
    without_dict = {}
    epcm_mean_df = exposure_df.ad_info.apply(lambda x: np.max([float(vallist.split(',')[-2]) for vallist in x.split(';')]))
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)
    test_ad_id_select['rival_max_epcm'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)
    #  每个广告对应所有请求曝光队列中广告被屏蔽概率的平均值的平均值
    without_dict = {}
    epcm_mean_df = exposure_df.ad_info.apply(lambda x: np.mean([float(vallist.split(',')[-1]) for vallist in x.split(';')]))
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)
    test_ad_id_select['rival_filter_rate'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)

    #  每个广告对应所有请求曝光队列中广告质量得分的平均值的平均值
    without_dict = {}
    epcm_mean_df = exposure_df.ad_info.apply(lambda x: np.mean([float(vallist.split(',')[-3]) for vallist in x.split(';')]))
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)
    test_ad_id_select['rival_q_epcm'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)

    #  每个广告对应所有请求曝光队列中广告点击率的平均值的平均值
    without_dict = {}
    epcm_mean_df = exposure_df.ad_info.apply(lambda x: np.mean([float(vallist.split(',')[-4]) for vallist in x.split(';')]))
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)
    test_ad_id_select['rival_pctr'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)

    # 每个广告对应所有请求曝光队列中广告数目的平均值
    without_dict = {}
    epcm_mean_df = exposure_df.ad_info.apply(lambda x: x.count(';')+1)
    epcm_mean_df.index = exposure_df.ad_user_time_id.astype(str)
    test_ad_id_select['rival_num'] = test_ad_id_select.requestion_list.progress_apply(get_rival_epcm)

    test_df = test_df.merge(test_ad_id_select[['ad_id', 'requestion_num', 'rival_epcm',
                                               'rival_max_epcm', 'rival_num', 'user_num',
                                               'rival_pctr', 'rival_q_epcm', 'rival_filter_rate']], on='ad_id', how='left')
    test_df.to_csv(output_path+'testset_cleaned_'+test_index+'.csv', index=False)
