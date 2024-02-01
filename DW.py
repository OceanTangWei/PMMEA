#-*-coding:GBK -*- 
import numpy as np
from scipy import sparse
import os
from ge import DeepWalk
import networkx as nx
from scipy.sparse import csgraph
import logging
#from cupyx.scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import random
logger = logging.getLogger(__name__)
#Full NMF matrix (which NMF factorizes with SVD)
#Taken from MILE code
import time

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


    
def netmf_mat_full(A, window = 5, b=1.0):
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    P,D_hat = get_degree_normalize(A,flag=False)
    #print "A shape", A.shape
    n = A.shape[0]
    vol = float(A.sum())
    D_hat = D_hat.astype(np.float32)
    P = P.astype(np.float32)
    S = P
    
    for i in range(window-1):
        #print "Compute matrix %d-th power" % (i + 1)
        
        S = P.dot(S) + P.todense()
        print(i)
    S *= vol / window / b
    
    M = D_hat.dot(S.T)
    M = M.T
    print("yes")
    result = np.log(np.maximum(M,1))
    print("yes")
    return sparse.csr_matrix(result)
def csr2coo(arr):
    coo = arr.tocoo()
    i = torch.as_tensor(cp.vstack((coo.row, coo.col)), dtype=torch.int64, device='cuda')
    v = torch.as_tensor(coo.data, device='cuda')
    arr_coo = torch.cuda.sparse.FloatTensor(i, v, torch.Size(coo.shape))
    return arr_coo
#Used in NetMF, AROPE
def svd_embed(prox_sim, dim):
#    prox_sim = cupyx.scipy.sparse.csr_matrix(prox_sim)
#    u, s, vh = svds(prox_sim, k=dim)
#    res = u @ cp.diag(s)
    
    
    #sklearn_svd = TruncatedSVD(dim, n_iter=5, random_state=100)
   
    #res = sklearn_svd.fit_transform(prox_sim)

    # 无语，sparse。linalg.svds居然有误差，同一个矩阵分解出来的差很多，SOS
    u, s, v = sparse.linalg.svds(prox_sim, dim, return_singular_vectors="u",tol=1e-3,maxiter=2)
    print("分解完毕")
    res = sparse.diags(np.sqrt(s)).dot(u.T).T

    # u,s,v = np.linalg.svd(prox_sim.todense(),dim)
    return res

def fast(SIM, train_nodes):
      SIM = SIM.todense()
 
      
      
      #train_nodes = random.sample([i for i in range(SIM.shape[0])], 300)
      C = SIM[:,train_nodes]
      W = C[train_nodes,:]
      W_pinv = np.linalg.pinv(W)
      U, X, V = np.linalg.svd(W_pinv)
      Wfac = np.dot(U, np.diag(np.sqrt(X)))
      res = np.dot(C, Wfac)
      return res

def netmf(A, dim = 128, window=5, b=1, normalize = False, train_nodes=None): #5,1
    
    prox_sim = netmf_mat_full(A, window, b)
    print("yes")
    sparsity = 1.0 - ( prox_sim.count_nonzero() / float(prox_sim.toarray().size) )
    print(sparsity)
    embed = svd_embed(prox_sim, dim)
    #embed = fast(prox_sim, train_nodes)
    if normalize:
        norms = np.linalg.norm(embed, axis = 1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    
    return embed


def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU
    
def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    
    M = X.dot(X.T)/(vol*b)
    
    result = np.log(np.maximum(M,1))
    
   
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(result))
    return sparse.csr_matrix(result)
def large_netmf(A,dim=128,window=5,b=1.0,normalize=False):
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=256, which="LA")
   
    s = time.time()
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
            window=window,
            vol=vol, b=b)
    use_time = time.time()-s
    e = time.time()
    print("matrix use time:{}".format(use_time))
    embed = svd_embed(deepwalk_matrix, dim)
    print("svd use time:{}".format(time.time()-e))
    if normalize:
        norms = np.linalg.norm(embed, axis = 1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    return embed
    
def sample_dw(adj, walk_length=80, num_walks=80,embed_size=128, window_size=5, workers=3, iters=5):
    
    path1 = "e1.npy"
    path2 = "e2.npy"
    if os.path.exists(path1):
      pos1 = np.load(path1)
      pos2 = np.load(path2)
    else:
      G = nx.from_scipy_sparse_matrix(adj)
      model = DeepWalk(G, walk_length=walk_length, num_walks=num_walks, workers=workers)
      model.train(window_size=window_size, embed_size=embed_size, iter=iters)
      embeddings = model.get_embeddings()
      
      emb_list = []
      
      for k in range(len(G)):
          emb_list.append(embeddings[k])
      emb_list = np.array(emb_list)
      pos1 = emb_list[::2,:]
      pos2 = emb_list[1::2,:] 
      np.save(path1, pos1)
      np.save(path2, pos2)
   
    return pos1,pos2
    
    
