import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch 
import numpy as np
import torch
import tqdm

from model_static_CEP import static_CEP_CP,static_CEP_Tucker_standard


data_file = '../processed_data/mvlens_small/mv_small_week_0.npy'



full_data = np.load(data_file, allow_pickle=True).item()

fold=0
R_U = 2

mode = 'CP'
# mode = 'Tucker'

# data_type = 'static'
data_type = 'dynamic'





# here should add one more data-loader class
data_dict = {}
data_dict['R_U'] = R_U


if data_type == 'static':
    data_dict['ndims'] = full_data['ndims'] 
    data_dict['ind_tr'] = full_data['train_ind']
    data_dict['ind_te'] = full_data['test_ind']

    
else:
    data_dict['ndims'] = full_data['ndims_time'] 
    data_dict['ind_tr'] = np.concatenate([full_data['train_ind'],full_data['train_time_disct'].reshape(-1,1)],1)
    data_dict['ind_te'] = np.concatenate([full_data['test_ind'],full_data['test_time_disct'].reshape(-1,1)],1)
    
if mode == 'CP':
    data_dict['gamma_size'] = data_dict['R_U']
else:
    data_dict['gamma_size'] = np.prod([R_U for k in range(len(data_dict['ndims']))])


data_dict['device'] = torch.device('cpu')
data_dict['v'] = 1
data_dict['v_time'] = 1

# data_dict['ind_tr'] = data_dict['tr_ind']#np.concatenate([data_dict['tr_ind'],data_dict['tr_T_disct'].reshape(-1,1)],1)
# data_dict['ind_te'] = data_dict['te_ind']#np.concatenate([data_dict['te_ind'],data_dict['te_T_disct'].reshape(-1,1)],1)


# data_dict['ind_tr'] = np.concatenate([data_dict['tr_ind'],data_dict['tr_T_disct'].reshape(-1,1)],1)
# data_dict['ind_te'] = np.concatenate([data_dict['te_ind'],data_dict['te_T_disct'].reshape(-1,1)],1)


data_dict['y_tr'] = torch.tensor(full_data['train_y']).double().reshape(-1,1)
data_dict['y_te'] =  torch.tensor(full_data['test_y']).double().reshape(-1,1)


data_dict['N'] = len(data_dict['ind_tr'])
data_dict['U'] = [torch.rand(dim,R_U).double() for dim in data_dict['ndims']]

# data_dict['R_U'] = R_U
# data_dict['gamma_size'] = 2

data_dict['a0'] = 1.0
data_dict['b0'] = 1.0

data_dict['DAMPPING_gamma']=0.5
data_dict['DAMPPING_U']=0.1

model = static_CEP_CP(data_dict)
model.E_gamma = torch.ones(model.gamma_size,1).double()
# model = static_CEP_Tucker_standard(data_dict)
EPOCH = 50
for i in tqdm.tqdm(range(EPOCH)):

    # gamma
    model.msg_update_gamma()
    model.post_update_gamma()
    model.expectation_update_gamma()

    # U
    model.msg_update_U()
    model.post_update_U() 
    model.expectation_update_z()
    for mode in range(model.nmod):
        model.expectation_update_z_del(mode)

    # tau
    model.msg_update_tau()
    model.post_update_tau()
    model.expectation_update_tau()

    if i % 2:
        loss_train,loss_test = model.model_test(data_dict['ind_te'],torch.tensor(data_dict['y_te']))
        print('loss_train: %.4f,loss_test_base: %.4f'%(loss_train,loss_test) )