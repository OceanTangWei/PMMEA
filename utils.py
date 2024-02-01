

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import faiss
import pickle
import json
from tqdm import tqdm
import tensorflow.keras.backend as K
from DW import netmf,large_netmf
import scipy.sparse as sp
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
import networkx as nx










def align_embedding(adj1 ,adj2 , adj_matrix, index1,index2, nodes1, nodes2,K_nei, dim=128,use_method="netmf"):




    path1 = "e1.npy"
    path2  = "e2.npy"
    if os.path.exists(path1):
      e1 = np.load(path1)
      e2 = np.load(path2)
    else:
      if use_method == "netmf":
        e1 = netmf(sp.csr_matrix(adj1),dim=dim,train_nodes=nodes1,window=5, b=1)
        e2 = netmf(sp.csr_matrix(adj2),dim=dim,train_nodes=nodes2,window=5, b=1)
      elif use_method == "large_netmf":
        e1 = large_netmf(sp.csr_matrix(adj1),dim=dim)
        e2 = large_netmf(sp.csr_matrix(adj2),dim=dim)
      else:
        e1, e2 = sample_dw(adj_matrix,embed_size=dim)
    np.save(path1, e1)
    np.save(path2, e2)
    e = np.concatenate((e1,e2),axis=0)
    e[index1,:] = e1
    e[index2,:] = e2


    obj = e[nodes1].T @  e[nodes2]

    e_star = e

    combined_e1 = [e1]
    combined_e2 = [e2]
 

    tmp1 = get_degree_normalize(adj1)
    tmp2 = get_degree_normalize(adj2)
    tmp = get_degree_normalize(adj_matrix)

 
    for i in range(K_nei):

        e_star = tmp @ e_star
        obj += e_star[nodes1].T @  e_star[nodes2]
 
    obj = obj/K_nei
    u, _, v = np.linalg.svd(obj)
    R = u @ v
    trans_e1 = e1 @ R
    e[index1,:] = trans_e1
 

    return e
    
def find_low_degree(adj_matrix, anchor_nodes, node_index, rel_index):
  triples = []
  g = nx.from_scipy_sparse_matrix(adj_matrix)
  degree_list = [(i, g.degree(i)) for i in node_index]
  tail_nodes = [item[0] for item in degree_list if item[1]<2]

  print(len(tail_nodes))
  for anchor_node in anchor_nodes:
    triples.extend([[tail_node, anchor_node, rel_index] for tail_node in tail_nodes if tail_node!= anchor_node])
    triples.extend([[anchor_node, tail_node, rel_index+1] for tail_node in tail_nodes if tail_node!= anchor_node])
 
  return triples
  
  
def load_graph_new(path,train_pairs):
    anchor_nodes1 = [item[0] for item in train_pairs[:5]]
    anchor_nodes2 = [item[1] for item in train_pairs[:5]]
    
        
    node_index1 = set()
    node_index2 = set() 
    triples = []
    with open(path + "triples_1") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
            node_index1.add(h)
            node_index1.add(t)
    with open(path + "triples_2") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
            node_index2.add(h)
            node_index2.add(t)
            
            
    triple_list = triples[:]
    node_index1 = sorted(list(node_index1))
    node_index2 = sorted(list(node_index2))
    triples = np.unique(triples,axis=0)
    node_size,rel_size = np.max(triples)+1 , np.max(triples[:,2])+1
    
    ent_tuple,triples_idx = [],[]
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()
    adj_matrix = sp.lil_matrix((node_size, node_size))
    for h,t,r in triples:
      adj_matrix[h, t] = 1
      adj_matrix[t, h] = 1
    last,index = (-1,-1), -1
    
    add_triples1 = find_low_degree(adj_matrix, anchor_nodes1, node_index1, rel_size)
    print("ADD TRIPLES1 :{}".format(len(add_triples1)))
    add_triples2 = find_low_degree(adj_matrix, anchor_nodes2, node_index2, rel_size)
    print("ADD TRIPLES2 :{}".format(len(add_triples2)))
    
    triple_list.extend(add_triples1)
    triple_list.extend(add_triples2)
    triples = np.unique(triple_list,axis=0)
    rel_size += 2
    
    adj_matrix = sp.lil_matrix((node_size, node_size))
    for i in range(node_size):
        ent_ent_s[(i,i)] = 0

    for h,t,r in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        
        ent_ent_s[(h,h)] += 1
        ent_ent_s[(t,t)] += 1

        if (h,t) != last:
            last = (h,t)
            index += 1
            ent_tuple.append([h,t])
            ent_ent_s[(h,t)] = 0

        triples_idx.append([index,r])
        ent_ent_s[(h,t)] += 1
        rel_ent_s.add((r,h))
        ent_rel_s.add((t,r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)
    
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, adj_matrix, ent_ent, ent_ent_val, rel_ent, ent_rel,node_index1, node_index2]
    pickle.dump(graph_data, open(path+"graph_cache.pkl","wb"))
    return graph_data
    
def load_graph(path):
    
    if os.path.exists(path+"graph_cache.pkl"):
        return pickle.load(open(path+"graph_cache.pkl","rb"))
        
        
    node_index1 = set()
    node_index2 = set() 
    triples = []
    with open(path + "triples_1") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
            node_index1.add(h)
            node_index1.add(t)
    with open(path + "triples_2") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
            node_index2.add(h)
            node_index2.add(t)
    node_index1 = sorted(list(node_index1))
    node_index2 = sorted(list(node_index2))
    triples = np.unique(triples,axis=0)
    node_size,rel_size = np.max(triples)+1 , np.max(triples[:,2])+1
    
    ent_tuple,triples_idx = [],[]
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()
    adj_matrix = sp.lil_matrix((node_size, node_size))
    last,index = (-1,-1), -1

    for i in range(node_size):
        ent_ent_s[(i,i)] = 0

    for h,t,r in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        
        ent_ent_s[(h,h)] += 1
        ent_ent_s[(t,t)] += 1

        if (h,t) != last:
            last = (h,t)
            index += 1
            ent_tuple.append([h,t])
            ent_ent_s[(h,t)] = 0

        triples_idx.append([index,r])
        ent_ent_s[(h,t)] += 1
        rel_ent_s.add((r,h))
        ent_rel_s.add((t,r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)
    
    print(node_size,rel_size)
    
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, adj_matrix, ent_ent, ent_ent_val, rel_ent, ent_rel,node_index1, node_index2]
    pickle.dump(graph_data, open(path+"graph_cache.pkl","wb"))
    return graph_data
    
    
def get_degree_normalize(adj, flag=True):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    if flag:
      return d_mat_inv_sqrt.dot(adj)
    else:
      return d_mat_inv_sqrt.dot(adj), d_mat_inv_sqrt
      
def load_aligned_pair(file_path,ratio = 0.3):
    with open(file_path + "ill_ent_ids") as f: #ref/ill
        ref = f.readlines()
    try:
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
    except:
        sup = None
    
    ref = np.array([line.replace("\n","").split("\t") for line in ref]).astype(np.int64)
    if sup:
        sup = np.array([line.replace("\n","").split("\t") for line in sup]).astype(np.int64)
        ref = np.concatenate([ref,sup])
    np.random.shuffle(ref)
    train_size = int(ref.shape[0]*ratio)
    return ref[:train_size],ref[train_size:]

def load_name_features(dataset,vector_path,mode = "word-level"):
    
    try:
        word_vecs = pickle.load(open("./word_vectors.pkl","rb"))
    except:
        word_vecs = {}
        with open(vector_path,encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                word_vecs[line[0]] = [float(x) for x in line[1:]]
        pickle.dump(word_vecs,open("./word_vectors.pkl","wb"))

    if "EN" in dataset:
        ent_names = json.load(open("./data/DBP15K/translated_ent_name/%s.json"%dataset[:-1].lower(),"r"))

    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((len(ent_names),300),"float32")
    char_vec = np.zeros((len(ent_names),len(d)),"float32")
    for i,name in tqdm(ent_names):
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word)-1):
                char_vec[i,d[word[idx:idx+2]]] += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d))-0.5
    
    faiss.normalize_L2(ent_vec)
    faiss.normalize_L2(char_vec)

    if mode == "word-level":
        name_feature = ent_vec
    if mode == "char-level":
        name_feature = char_vec
    if mode == "hybrid-level": 
        name_feature = np.concatenate([ent_vec,char_vec],-1)
        
    return name_feature

def sparse_sinkhorn_sims(left,right,features,top_k=500,iteration=15,mode = "test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l); faiss.normalize_L2(features_r)

    res = faiss.StandardGpuResources()
    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    return K.reshape(row_index[:,1],(-1,top_k)), K.reshape(row_sims,(-1,top_k))

def test(test_pair,features,top_k=500,iteration=15):
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    ranks = tf.argsort(-sims,-1).numpy()
    index = index.numpy()
    
    wrong_list,right_list = [],[]
    h1,h10,mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i],right[index[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n"%(h1/len(test_pair),h10/len(test_pair),mrr/len(test_pair)))
    
    return right_list, wrong_list