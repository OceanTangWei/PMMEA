# %%

# choose the GPU, "-1" represents using the CPU

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# import all the requirements
import faiss
from utils import *
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from  utils import align_embedding
from read_aux_features import load_aux_features
import math
from scipy import sparse
from sklearn.preprocessing import normalize
from DW import svd_embed
import time
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %%

from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
def fast_factorize(feature):
  SIM  = cosine_similarity(feature, feature)**3
  
  return svd_embed(SIM, 512)
  

# choose the dataset and set the random seed
# the first run may be slow because the graph needs to be preprocessed into binary cache
dataname = "fb_db"
np.random.seed(12306) #12306

dataset = dataname + "/"


path = "./data/DBP15K/" + dataset
# set hyper-parameters, load graphs and pre-aligned entity pairs
# if your GPU is out of memory, try to reduce the ent_dim

ent_dim, depth, top_k = 1024, 2, 500
if "EN" in dataset:
    rel_dim, mini_dim = ent_dim // 2, 16# 16
else:
    rel_dim, mini_dim = ent_dim // 3, 16 #16
pos_dim = 512


train_pair, test_pair = load_aligned_pair(path, ratio=0.3)

node_size, rel_size, ent_tuple, triples_idx, adj_matrix, ent_ent, ent_ent_val, rel_ent, ent_rel,node_index1, node_index2 = load_graph(path)



candidates_x, candidates_y = set([x for x, y in test_pair]), set([y for x, y in test_pair])

# %%
def pad(x, target_dim):
  n, dim = x.shape
  diff = target_dim-dim
  if diff == 0:
    return x
  padding = np.zeros((n,diff))
  res = np.concatenate((x,padding),axis=-1)
  return res


img_features, name_features, char_features,att_features,rel_features   = load_aux_features(file_dir="data/DBP15K/{}".format(dataname))



img_features = img_features.astype(np.float32)
att_features =att_features.astype(np.float32)



adj1=adj_matrix[node_index1,:][:,node_index1]

adj2=adj_matrix[node_index2,:][:,node_index2]


node_list1 = [item[0] for item in train_pair]
node_list2 = [item[1] for item in train_pair]

start_time = time.time()
pos = align_embedding(adj1 ,adj2 , adj_matrix, node_index1, node_index2, node_list1, node_list2, K_nei=7,dim=pos_dim,use_method="netmf")


pos_img = pad(pos, img_features.shape[-1])
pos_att = pad(pos, att_features.shape[-1])
pos_rel = pad(pos, rel_features.shape[-1])
pos_name = pos




ent_pos = pad(pos, ent_dim)



def random_projection(x, out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1], out_dim)), axis=-1)
    return K.dot(x, random_vec)

def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=128, save_mem=False):
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor[:, i * batch_size:(i + 1) * batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return K.concatenate(results, -1)


def get_features(train_pair, pos_feature, extra_feature=None,alpha=1e-2):
    random_vec = K.l2_normalize(tf.random.normal((len(train_pair), pos_feature.shape[-1]),mean=0.0,stddev=1.0)**1.0, axis=-1)
    pos_feature = tf.tensor_scatter_nd_update(tf.convert_to_tensor(pos_feature), train_pair.reshape((-1, 1)),
                                                  tf.repeat(random_vec, 2, axis=0))
    pos_feature = tf.nn.l2_normalize(pos_feature, axis=-1)
    if extra_feature is not None:
        ent_feature = extra_feature
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair), extra_feature.shape[-1]),mean=0.0,stddev=1.0)**1.0, axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.convert_to_tensor(extra_feature), train_pair.reshape((-1, 1)),
                                                  tf.repeat(random_vec, 2, axis=0))
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair), ent_dim),mean=0.0,stddev=1.0)**1.0, axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size, ent_dim)), train_pair.reshape((-1, 1)),
                                                  tf.repeat(random_vec, 2, axis=0))
    rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))

    ent_ent_graph = tf.SparseTensor(indices=ent_ent, values=ent_ent_val, dense_shape=(node_size, node_size))
    rel_ent_graph = tf.SparseTensor(indices=rel_ent, values=K.ones(rel_ent.shape[0]), dense_shape=(rel_size, node_size))
    ent_rel_graph = tf.SparseTensor(indices=ent_rel, values=K.ones(ent_rel.shape[0]), dense_shape=(node_size, rel_size))
    

    ent_list, rel_list, pos_list = [ent_feature], [rel_feature], [pos_feature]

    beta = (1-alpha)/2.0

    

    for i in range(2):
     
        
        new_rel_feature = math.sqrt(1-alpha)*batch_sparse_matmul(rel_ent_graph, ent_feature) + batch_sparse_matmul(rel_ent_graph, pos_feature)*math.sqrt(alpha)
        
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)               

        new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)*math.sqrt(beta)
        new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)*math.sqrt(beta)
        new_ent_feature = new_ent_feature + batch_sparse_matmul(ent_ent_graph, pos_feature) *math.sqrt(alpha)  
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

        ent_feature = new_ent_feature;
        rel_feature = new_rel_feature
        ent_list.append(ent_feature);
        rel_list.append(rel_feature)

    ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
    rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
    rel_feature = random_projection(rel_feature, rel_dim)

    batch_size = ent_feature.shape[-1] // mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx, values=K.ones(triples_idx.shape[0]),
                                   dense_shape=(np.max(triples_idx) + 1, rel_size))
    adj_value = batch_sparse_matmul(sparse_graph, rel_feature)

    features_list = []
    for batch in range(rel_dim // batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch * batch_size + head >= rel_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                           dense_shape=(node_size, node_size))
            feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list, -1).numpy())

    features = np.concatenate(features_list, axis=-1)

    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature, features], axis=-1)
    return features

#def get_top_k(mat,topk):
#  x = np.argpartition(-mat, kth=2, axis=-1)
#  index = x[:,:topk]
#  y = np.zeros_like(x)
#  node_list = list(range(mat.shape[0]))
#  for i in range(topk):
#    y[node_list,index[:,i]] = x[node_list,index[:,i]]
#  return normalize(y)


  
  
  


epochs = 1000
alpha = 0.1 



for epoch in range(epochs):
    
    print("Round %d start:" % (epoch + 1))
    
    
    using_name_features = True

    if using_name_features and "EN" in dataset:
        
    
        new_img_features = get_features(train_pair,pos_img.astype(np.float32), extra_feature=img_features,alpha=alpha)
        new_att_features = get_features(train_pair,pos_att.astype(np.float32), extra_feature=att_features,alpha=alpha)

        l_features = np.concatenate([new_img_features,new_att_features], -1)

    

    
    s_features = get_features(train_pair, ent_pos.astype(np.float32),alpha=alpha)
    if using_name_features and "EN" in dataset:
        features = np.concatenate([s_features, l_features], -1)
    else:
        features = s_features

    print(features.shape)
    thsh = 0.7
    if epoch < epochs - 1:
        thsh = thsh-0.02*epoch
        left, right = list(candidates_x), list(candidates_y)
    
                
        index, sims = sparse_sinkhorn_sims(left, right, features, top_k)
        ranks = tf.argsort(-sims, -1).numpy()
        sims = sims.numpy();
        index = index.numpy()
        
        temp_pair = []
        x_list, y_list = list(candidates_x), list(candidates_y)
   
        
        for i in range(ranks.shape[0]):
            
            if sims[i, ranks[i, 0]] > thsh:
                x = x_list[i]
                y = y_list[index[i, ranks[i, 0]]]
                temp_pair.append((x, y))

        for x, y in temp_pair:
            if x in candidates_x:
                candidates_x.remove(x);
            if y in candidates_y:
                candidates_y.remove(y);
        
        print(time.time()-start_time)
        print("new generated pairs = %d" % (len(temp_pair)))
        print("rest pairs = %d" % (len(candidates_x)))

        if not len(temp_pair):
            break
        train_pair = np.concatenate([train_pair, np.array(temp_pair)])
        
        
    node_list1 = [item[0] for item in train_pair]
    node_list2 = [item[1] for item in train_pair]
    pos = align_embedding(adj1 ,adj2 , adj_matrix, node_index1, node_index2, node_list1, node_list2, K_nei=7)
   
   
    right_list, wrong_list = test(test_pair, features, top_k=1000)



