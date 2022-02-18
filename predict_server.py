'''
预估服务部分：
从pd中加载模型；load_model return sess, input_tensor, output_tensor
请求特征服务器获得input；
基于特征值去参数服务器获取参数，如embedding(note:也在不断更新，不同于vector_server中)
输入模型，返回结果
'''

import random 
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import numpy as np

pb_path = 'data/deepfm/saved_pd'
feature_embedding_path = 'data/saved_deepfm_embedding'
batch_size = 16
feature_per_ui = 8
feature_embed_dimension = 5

def load_model():
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(sess,[tag_constants.SERVING],pb_path)
    
    signature = meta_graph_def.signature_def
    in_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['input0'].name
    out_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['output0'].name

    in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
    out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

    return sess, in_tensor, out_tensor

def load_emb():
    hashcode_dict = {}
    with open(feature_embedding_path, 'r') as lines:
        for line in lines:
            tmp = line.strip().split('\t')
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(',')]
                hashcode_dict[tmp[0]] = vec
    return hashcode_dict

# 模拟实际request user+id 对
def get_request():
    embed_dict = load_emb()
    keys = embed_dict.keys()
    hashcode = []
    batch_sample = []
    for _ in range(batch_size):
        feature_value = random.sample(keys,8)
        hashcode.append(','.join([str(_) for _ in feature_value]))  # 一个ui对
        hashcode_embed = []
        for s in feature_value:
            hashcode_embed.append(embed_dict[s])
        batch_sample.append(hashcode_embed)
    return hashcode, batch_sample

def batch_predict( ):
    requests, batch_sample = get_request()
    sess, in_tensor, out_tensor = load_model()
    predict = sess.run(out_tensor,feed_dict = {in_tensor: np.array(batch_sample)})
    for request,predict_score in zip(requests, predict):
        print(request + '\t'+ str(predict_score[0]))

if __name__=='__main__':
    batch_predict()