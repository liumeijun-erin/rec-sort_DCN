# -*- coding: utf-8 -*-
'''
读取原数据生成特征并保存
用户特征 db1
商品特征 db2
'''
import redis
import traceback
import os
import pandas as pd
import json

def save_redis(items, db = 1):
    redis_url = 'redis://:@127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()

def extract_feature():
    feature_names = ['item_id', 'user_id', 'user_session', 'user_type',\
                    'user_prefer1', 'user_prefer2', 'user_prefer3', 'user_env',\
                    'label']
    raw_data_path = 'data'
    files_list = []
    for file in os.listdir(raw_data_path):
        files_list.append(os.path.join(raw_data_path, file))
    
    files_list.sort()  # pre：文件名按时间排列
    ds_list = []
    for f in files_list:
        ds_list.append(pd.read_csv(f, header=None, names= feature_names))
    ds = pd.concat(ds_list, axis = 0, ignore_index = True)
    ds = ds.astype(str)
    print(ds.head())

    user_feature_dict = {}
    # 遍历目的：获取最新数据
    for user_info in zip(ds['user_id'], ds['user_session'], ds['user_type'], \
        ds['user_prefer1'], ds['user_prefer2'], ds['user_prefer3'], ds['user_env']):
        user_feature_dict[user_info[0]] = json.dumps({
            'user_id': user_info[0],
            'user_session': user_info[1],
            'user_type': user_info[2],
            'user_prefer1': user_info[3],
            'user_prefer2': user_info[4],
            'user_prefer3': user_info[5],
            'user_env': user_info[6]
        })
    save_redis(user_feature_dict.items(), 1)

    item_feature_dict = {}
    for item_info in zip(ds['item_id']):
        item_feature_dict[item_info[0]] = json.dumps({
            'item_id': item_info[0]
        })
    save_redis(item_feature_dict.items(), 2)


if __name__ == '__main__':
    extract_feature()