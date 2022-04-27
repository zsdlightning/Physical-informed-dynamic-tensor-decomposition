import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch 
import numpy as np
import torch
# from torch.optim import Adam
import tqdm
from model_ADF_static import static_ADF


import utils
# import data_loader
import time

data_file = '../processed_data/beijing_15k.npy'
# data_file = '../processed_data/mvlens_10k.npy'
# data_file = '../processed_data/server_10k.npy'
# data_file = '../processed_data/dblp_50k.npy'
# data_file = '../processed_data/ctr_10k.npy'

full_data = np.load(data_file, allow_pickle=True).item()

fold=0
R_U = 2

# mode = 'static'
mode = 'discret_time'  # add the discrete timestamp as extra mode


# here should add one more data-loader class
data_dict = full_data['data'][fold]
data_dict['device'] = torch.device('cpu')


if mode == 'discret_time':
    data_dict['ndims'] = full_data['ndims']+ [len(full_data['time_uni'])]
    data_dict['tr_ind'] = np.concatenate([data_dict['tr_ind'],data_dict['tr_T_disct'].reshape(-1,1)],1)
    data_dict['te_ind'] = np.concatenate([data_dict['te_ind'],data_dict['te_T_disct'].reshape(-1,1)],1)

else:
    data_dict['ndims'] = full_data['ndims']
    data_dict['tr_ind'] = data_dict['tr_ind']
    data_dict['te_ind'] = data_dict['te_ind']

data_dict['num_node'] = full_data['num_node']

# data_dict['time_id_table'] = full_data['time_id_table']
# data_dict['time_uni'] = full_data['time_uni']

hyper_dict = {}
hyper_dict['device'] = torch.device("cpu")
hyper_dict['R_U'] = 2 # dim of each node embedding
hyper_dict['c'] = 0.1 # diffusion rate
hyper_dict['a0']=1.0
hyper_dict['b0']=1.0
hyper_dict['DAMPING']=0.5


EPOCH = 10

model =static_ADF(data_dict, hyper_dict)

# N_T = len(data_dict['time_uni'])
# for epoch in tqdm.tqdm(range(EPOCH)):
#     for T in range(N_T):
#         model.ADF_update_T(T)\
        
#         if T % 100 ==0:
#             test_rmse = model.model_test()
#             print('it: %d, T: %d test_rmse: %.4f '%(epoch,T,test_rmse)) 

N = len(data_dict['tr_ind'])
for epoch in tqdm.tqdm(range(EPOCH)):
    for n in range(N):
        model.ADF_update_N(n)
        
        if n % 2000 ==0:
            test_rmse = model.model_test()
            print('it: %d, n: %d test_rmse: %.4f '%(epoch,n,test_rmse)) 
