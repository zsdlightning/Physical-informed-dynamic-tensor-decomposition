import numpy as np
import torch
import utils
import scipy
from scipy import linalg
from torch.utils.data import Dataset

def build_time_data_table(time_ind):
    # input: sorted time-stamp seq (duplicated items exists) attached with data seq 
    # output: table (list) of associated data points of each timestamp
    # ref: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function/43094244
    # attention, here the input "time-stamps" can be either (repeating) id, or exact values, but seq length must match data seq
    # in out table, order of item represents the time id in order 
    time_data_table = np.split(np.array([i for i in range(len(time_ind))]),np.unique(time_ind,return_index=True)[1][1:])
    return time_data_table


def build_id_key_table(nmod,ind):
    # build uid-data_key_table, implement by nested list
    
    # given indices of unique rows of each mode/embed (store in uid_table)  
    uid_table = []
    
    # we could index which data points are associated through data_table
    data_table = []

    for i in range(nmod):
        values,inv_id = np.unique(ind[:,i],return_inverse=True)
        uid_table.append(list(values))

        sub_data_table = []
        for j in range(len(values)):
            data_id = np.argwhere(inv_id==j)
            if len(data_id)>1:
                data_id = data_id.squeeze().tolist()
            else:
                data_id = [[data_id.squeeze().tolist()]]
            sub_data_table.append(data_id)
            
        data_table.append(sub_data_table)
        
    return uid_table,data_table

def generate_mask(ndims,ind):
    num_node = sum(ndims)
    nmod = len(ndims)
    ind = torch.tensor(ind)

    mask = torch.zeros((num_node, num_node))
    for i in range(1, nmod):
        row = np.sum(ndims[:i])
        for j in range(i):
            col = np.sum(ndims[:j])
            indij = ind[:, [i, j]]
            indij = torch.unique(indij, dim=0).long()
            row_idx = row + indij[:, 0]
            col_idx = col + indij[:, 1]
            mask[row_idx.long(), col_idx.long()] = 1
    return mask


def generate_Lapla(ndims,ind):
    '''
    generate the fixed Laplacian mat of prior K-partition graph,
    which is defined by the observed entries in training set    
    '''
    num_node = sum(ndims)

    W_init = torch.ones((num_node, num_node))
    mask = generate_mask(ndims,ind)
    Wtril = torch.tril(W_init)*mask
    W = Wtril + Wtril.T
    D = torch.diag(W.sum(1))
    return W-D

def generate_state_space_Matern_23(data_dict,hyper_dict):
    '''
    For matern 3/2 kernel with given hyper-paras and data,
    generate the parameters of coorspoding state_space_model,
    recall: for each dim of all-node-embedding, the form of state_space_model is iid (independent & identical)
    
    input: data_dict, hyper_dict 
    output: trans mat: F,  stationary covarianc: P_inf

    '''

    ndims = data_dict['ndims']
    D = data_dict['num_node']
    ind = data_dict['tr_ind']

    # hyper-para of kernel
    lengthscale = hyper_dict['ls']
    variance = hyper_dict['var']
    c = hyper_dict['c'] # diffusion rate 
    
    lamb = np.sqrt(3)/lengthscale
    
    # F = torch.zeros((2*D, 2*D), device=data_dict['device'])
    F = np.zeros((2*D, 2*D))
    F[:D,:D] = utils.generate_Lapla(ndims,ind)*c
    F[:D,D:] = np.eye(D)
    F[D:,:D] = -np.square(lamb) * np.eye(D)
    F[D:,D:] = -2 * lamb *  np.eye(D)

    Q_c = 4 * lamb**3 * variance * np.eye(D)
    L = np.zeros((2*D, D)) 
    L[D:,:] = np.eye(D)
    Q = - np.matmul(np.matmul(L,Q_c),L.T)
    

    P_inf = Lyapunov_slover(F,Q)

    return torch.tensor(F,device=hyper_dict['device']), torch.tensor(P_inf,device=hyper_dict['device'])


def Lyapunov_slover(F,Q):
    '''
    For the given mix-process SDE, solve correspoding Lyapunov to get P_{\inf}  
    '''
    
    return linalg.solve_continuous_lyapunov(F, Q)