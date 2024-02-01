
from load import *

from sklearn.preprocessing import normalize

import os






def load_img_features(ent_num, file_dir):
    # load images features
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    elif "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = "data/mmkb-datasets/" + filename + "/" + filename + "_id_img_feature_dict.pkl"
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"

    img_features = load_img(ent_num, img_vec_path)
    return img_features


def load_aux_features(file_dir="data/DBP15K/zh_en", word_embedding="glove"):
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    print("total ent num: {}, rel num: {}".format(ENT_NUM, REL_NUM))
    
    img_features = load_img_features(ENT_NUM, file_dir)
    img_features = normalize(img_features)
    print("image feature shape:", img_features.shape)
    
    rel_features = load_relation(ENT_NUM, triples, 1000)
    #rel_features = normalize(rel_features)
    print("relation feature shape:", rel_features.shape)
    
    
    
    # load name/char features (only for DBP15K datasets)
    data_dir, dataname = os.path.split(file_dir)
    if word_embedding == "glove":
      word2vec_path = "data/embedding/glove.6B.300d.txt"
    elif word_embedding == 'fasttext':
      pass
    else:
      raise Exception("error word embedding")


    name_features = None
    char_features = None
    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # attr
    att_features = normalize(att_features)
 
    print("attribute feature shape:", att_features.shape)
    
    return img_features, name_features, char_features, att_features, rel_features

