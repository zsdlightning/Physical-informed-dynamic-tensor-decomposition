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

def generate_mask(data_dict):
    num_node = data_dict['num_node']
    device = data_dict['device']
    nmod = data_dict['nmod']

    mask = torch.zeros((num_node, num_node), device=device)
    for i in range(1, nmod):
        row = np.sum(data_dict['ndims'][:i])
        for j in range(i):
            col = np.sum(data_dict['ndims'][:j])
            indij = data_dict['ind'][:, [i, j]]
            indij = torch.unique(indij, dim=0).long()
            row_idx = row + indij[:, 0]
            col_idx = col + indij[:, 1]
            mask[row_idx.long(), col_idx.long()] = 1
    return mask


def generate_Lapla(data_dict):
    '''
    generate the fixed Laplacian mat of prior K-partition graph,
    which is defined by the observed entries in training set    
    '''

    W_init = torch.ones((data_dict['num_node'], data_dict['num_node']), device=data_dict['device'])
    mask = generate_mask(data_dict)
    Wtril = torch.tril(W_init)*mask
    W = Wtril + Wtril.T
    D = torch.diag(W.sum(1))
    return W-D