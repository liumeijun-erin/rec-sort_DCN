# -*- coding: utf-8 -*-

'''
model模块中数据流预处理部分:
读取数据，划分数据集; 定义哈希 + 将所有特征项进行hash(名+值); 转换为tfrecords类型
'''
import os
import pandas as pd
import tensorflow as tf
print(tf.__version__)

def bkdt2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)
    return hash & mask60

def split_train_test(raw_data_path):
    feature_names = ['item_id', 'user_id', 'user_session', 'user_type',\
                    'user_prefer1', 'user_prefer2', 'user_prefer3', 'user_env',\
                    'label']
    
    files_list = []
    for file in os.listdir(raw_data_path):
        files_list.append(os.path.join(raw_data_path, file))
    
    files_list.sort()
    train_data_files = files_list[0:-1]
    test_data_file = files_list[-1]

    train_ds_list = []
    for file in train_data_files:
        train_ds_list.append(pd.read_csv(file, header = None, names=feature_names))

    train_ds = pd.concat(train_ds_list, axis=0, ignore_index=True)
    train_ds = train_ds.astype(str)
    # print(train_ds.head())

    test_ds_list = []
    test_ds = pd.read_csv(test_data_file, header = None, names = feature_names)
    test_ds = test_ds.astype(str)
    # print(test_ds.head())
    return train_ds, test_ds

def tohash(data, save_path):
    wf = open(save_path, 'w')
    for line in data.values:
        item_id = bkdt2hash64('item_id='+str(line[0]))
        user_id = bkdt2hash64('user_id='+str(line[1]))
        user_session = bkdt2hash64('user_session='+str(line[2]))
        user_type = bkdt2hash64('user_type='+str(line[3]))
        user_prefer1 = bkdt2hash64('user_prefer1='+str(line[4]))
        user_prefer2 = bkdt2hash64('user_prefer2='+str(line[5]))
        user_prefer3 = bkdt2hash64('user_prefer3='+str(line[6]))
        user_env = bkdt2hash64('user_env='+str(line[7]))
        
        wf.write(str(item_id) + ',' + str(user_id) + ',' \
            + str(user_session) + ',' + str(user_type) + ',' \
            + str(user_prefer1) + ',' + str(user_prefer2) + ',' \
            + str(user_prefer3) + ',' + str(user_env) + ',' + str(line[8]) + '\n')
    wf.close()     

def get_tfrecords_example(feature, label):
    tfrecords_features = {
        'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=tfrecords_features)
    )

def totfrecords(file, save_dir):
    print("Generating tfrecords from file: %s..." % file)
    file_num = 0
    writer = tf.python_io.TFRecordWriter(save_dir + '/' + 'part-%06d'%file_num + '.tfrecords')
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = line.strip().split(',')
            feature = [int(tmp[0]),int(tmp[1]),int(tmp[2]),int(tmp[3]),\
                    int(tmp[4]),int(tmp[5]),int(tmp[6]),int(tmp[7])]
            label = [float(tmp[8])]
            example = get_tfrecords_example(feature, label)
            writer.write(example.SerializeToString())
            if (i+1) % 20000 == 0:
                writer.close()
                file_num += 1
                writer = tf.python_io.TFRecordWriter(save_dir + '/' + 'part-%06d' % file_num + '.tfrecords')
    writer.close()

if __name__ == '__main__':
    train_hash_path = 'data/hash_data/train_hash'
    test_hash_path = 'data/hash_data/test_hash'
    if not os.path.exists('data/hash_data'):
        train_ds, test_ds = split_train_test('data/')
        os.mkdir('data/hash_data')
        tohash(train_ds, train_hash_path)
        tohash(test_ds, test_hash_path)
        print("Hash process finished.")

    train_tfrecords_path = 'data/train'
    test_tfrecords_path = 'data/test'
    if not os.path.exists(train_tfrecords_path):
        os.mkdir(train_tfrecords_path)
        totfrecords(train_hash_path, train_tfrecords_path)
    print("train tfrecords files generated.")
    if not os.path.exists(test_tfrecords_path):
        os.mkdir(test_tfrecords_path)
        totfrecords(test_hash_path, test_tfrecords_path)
    print("test tfrecords files generated.")