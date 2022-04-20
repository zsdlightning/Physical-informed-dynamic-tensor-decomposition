import numpy as np
import scipy
import pandas
import torch
import utils
from utils import generate_state_space_Matern_23
from scipy import linalg
from utils import build_id_key_table

# data process:

class Bayes_diffu_tensor():
    def __init__(self,data_dict,hyper_dict):


        # hyper-paras
        self.epoch = hyper_dict['epoch'] # passing epoch
        self.R_U = hyper_dict['R_U'] # rank of latent factor of embedding
        self.device = hyper_dict['device']
        
        self.a0 = hyper_dict['a0']
        self.b0 = hyper_dict['b0']
        
        

        # data-dependent paras
        self.data_dict = data_dict
        
        self.ind_tr = data_dict['tr_ind']
        self.y_tr = data_dict['tr_y'].to(self.device) # N*1
        self.N = len(data_dict['y_tr'])
        self.ndims = data_dict['ndims']
        self.nmod = len(self.ndims)
        self.num_nodes = sum(self.ndims)
        
        self.train_time_ind =data_dict['tr_T_disct'] # N*1
        self.test_time_ind = data_dict['te_T_disct'] # N*1
        
        self.time_uni = data_dict['time_uni'] # N_time*1
        self.N_time = data_dict['N_time']  
        
        self.time_id_table = data_dict['time_id_table']
        self.F = data_dict['F'].to(self.device) # transition matrix
        self.P_inf = data_dict['P_inf'].to(self.device) # stationary covar
        
    
            
        # init the message factor of llk term (U_llk, tau)
        # and transition term (U_f: U_forard, U_b: U_backward)
        
        # actually, it's the massage from llk-factor -> variabel U
        
        self.msg_U_llk_m = torch.rand(self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        self.msg_U_llk_v = torch.ones(self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        
        # self.msg_U_llk_m = [torch.rand(ndim,self.R_U,self.N_time).double().to(self.device) \
        #     for ndim in self.ndims] 
        
        self.msg_U_llk_v = [torch.ones(ndim,self.R_U,self.N_time).double().to(self.device) \
            for ndim in self.ndims] 
        
        self.msg_tau_a = torch.ones(self.N_time,1).to(self.device)
        self.msg_tau_b = torch.ones(self.N_time,1).to(self.device)
        
        # actually, it's the massage from transition-factor -> variabel U
        # for here, we arrange the U-msg by concat-all-as-tensor for efficient computing in transition
        # recall, with Matern 23 kernel, msg_U_transition = [ U, U'], so the firsr-dim is 2*num_nodes 
        
        self.msg_U_f_m = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device) 
        self.msg_U_f_v = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        
        self.msg_U_b_m = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        self.msg_U_b_v = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)   


    
        # init the calibrating factors / q_del in draft, init/update with current msg
        
        # actually, it's the massage from variabel U -> llk-factor
        self.msg_U_llk_m_del = None
        self.msg_U_llk_v_del = None
          
        self.msg_tau_a_del_T = None
        self.msg_tau_b_del_T = None
                
        # actually, it's the massage from variabel U -> trans-factor
        self.msg_U_f_m_del = None
        self.msg_U_f_v_del = None
        
        self.msg_U_b_m_del = None
        self.msg_U_b_v_del = None
        
        # as the computing of msg_tau_del is trival, we don't assign extra varb for it    
        
        # init the post. U
        
        self.post_U_m = [torch.rand(ndim,self.R_U,self.N_time).double().to(self.device) \
            for ndim in self.ndims]
        
        self.post_U_v = [torch.ones(ndim,self.R_U,self.N_time).double().to(self.device) \
            for ndim in self.ndims] 
        
        # time-data table 
        # Given a time id, return the indexed of entries  
        # self.uid_table, self.data_table = build_id_key_table(self.nmod,self.ind_tr)
        self.time_data_table_tr = utils.build_time_data_table(self.train_time_ind) 
        self.time_data_table_te = utils.build_time_data_table(self.test_time_ind) 
        
        
        self.ind_T = None
        self.y_T = None
        
        self.Z_T = None
        self.Z_2_T = None
        

def msg_update_U_llk(self,T):
    
    # retrive the observed entries at T
    eind_T = self.time_data_table_tr[T] # id of obseved entries at this time-stamp
    N_T = len(eind_T) 
    ind_T = self.ind_tr[eind_T]
    y_T = self.y_tr[eind_T]
    
    
    
    uid_table, _ = build_id_key_table(self.nmod,ind_T) # get the id of associated nodes
    self.uid_table_T = uid_table
    self.ind_T = ind_T
    
    U_llk_del_T = torch.cat([self.msg_U_llk_m_del[:,:,T],self.msg_U_llk_v_del[:,:,T]]) # concat the m and v 
    U_llk_del_T.requires_grad=True 
    
    U_llk_del_T_m,U_llk_del_T_v = self.arrange_U_llk(U_llk_del_T) # arrange U as mode-wise
    
    E_z_del, E_z_2_del = self.moment_product_U_del(U_llk_del_T_m,U_llk_del_T_v) # first and second moment of 
    E_tau_del = self.msg_tau_a_del_T[T]/self.msg_tau_b_del_T[T]
    
    log_zn = self.ADF_update_llk(U_llk_del_T)
    

def ADF_update_llk(self,U_llk_del_T):
     
    grads = torch.autograd.functional.jacobian(self.log_Z_llk, U_llk_del_T)     
    

    
def arrange_U_llk(self,U_llk_del_T):
    # arrange_U_to_mode-wise for convinience in computing
    U_llk_del_T_m = U_llk_del_T[:self.num_nodes,:]
    U_llk_del_T_v = U_llk_del_T[self.num_nodes:,:]
    
    # arrange_U_to_mode-wise for convinience in computing
    U_llk_del_T_m = []
    U_llk_del_T_v = []
    
    idx_start = 0     
        
    for ndim in self.ndims:
        idx_end = idx_start + ndim
        U_llk_del_T_mode_m = U_llk_del_T[idx_start:idx_end,:]
        U_llk_del_T_mode_v = U_llk_del_T[self.num_nodes+idx_start:self.num_nodes+idx_end,:]
        
        
        U_llk_del_T_m.append(U_llk_del_T_mode_m)
        U_llk_del_T_v.append(U_llk_del_T_mode_v)
        
        idx_start = idx_end
        
    return 
    
    
def moment_product_U_del(self,U_llk_T_m,U_llk_T_v):
    # compute first and second moments of \Hadmard_prod_{k \in given modes} u_k -CP based on the U_llk_del
    # based on the U_llk_del (calibrating factors)
    E_z = U_llk_T_m[0][self.ind_T[:,0]] # N*R_u*1
    E_z_2 = U_llk_T_v[0][self.ind_T[:,0]] + E_z * E_z # N*R_u*1

    for mode in range(1,self.nmod):
        E_u = U_llk_T_m[mode][self.ind_T[:,mode]] # N*R_u*1
        E_u_2 = U_llk_T_v[mode][self.ind_T[:,mode]] + E_u * E_u # N*R_u*1

        E_z = E_z*E_u
        E_z_2 = E_z_2*E_u_2  
        
    return E_z, E_z_2
  
    
def msg_update_U_trans(self,T,mode='forward'):
    pass

def msg_update_tau(self,a,b,T):
    self.msg_tau_a[T] = a
    self.msg_tau_b[T] = b
    
def msg_update_tau_del(self,T):
    self.msg_tau_a_del_T[T] = self.msg_tau_a[:T].sum() + self.msg_tau_a[T+1:].sum() - (self.N_time-1)
    self.msg_tau_b_del_T[T] = self.msg_tau_b[:T].sum() + self.msg_tau_b[T+1:].sum()

def msg_update_U_llk_del(self,T):
    # with message-passing framework, we just merge in-var-msg from all branches to get q_del
    # no need to compute from posterior/cur-factor
    # U_llk_del = U_f + U_b
    
    msg_U_llk_del_v_inv = \
                1.0/self.msg_U_f_v[:self.num_nodes,:,T] + 1.0/self.msg_U_b_v[:self.num_nodes,:,T] 
    
    msg_U_llk_del_v_inv_m = \
                torch.div(self.msg_U_f_m[:self.num_nodes,:,T],self.msg_U_f_v[:self.num_nodes,:,T])\
                +torch.div(self.msg_U_b_m[:self.num_nodes,:,T],self.msg_U_b_v[:self.num_nodes,:,T])
    
    self.msg_U_llk_v_del[:,:,T] = 1.0/msg_U_llk_del_v_inv
    self.msg_U_llk_m_del[:,:,T] = (1.0/msg_U_llk_del_v_inv) * msg_U_llk_del_v_inv_m
                            
    
def msg_update_U_trans_del(self,T):
    # U_f_del = U_b + U_llk
    # U_b_del = U_f + U_llk
    
    # forward
    msg_U_f_del_v_inv = \
        1.0/self.msg_U_b_v[:self.num_nodes,:,T] + 1.0/self.msg_U_llk_v[:,:,T]
    msg_U_f_del_v_inv_m = \
        torch.div(self.msg_U_b_m[:self.num_nodes,:,T],self.msg_U_b_v[:self.num_nodes,:,T])\
        +torch.div(self.msg_U_llk_m[:,:,T],self.msg_U_llk_v[:,:,T])
        
    self.msg_U_f_v_del[:self.num_nodes,:,T] = 1.0/msg_U_f_del_v_inv
    self.msg_U_f_v_del[self.num_nodes:,:,T] = self.msg_U_b_v[self.num_nodes:,:,T]
    
    self.msg_U_f_m_del[:self.num_nodes,:,T] = (1.0/msg_U_f_del_v_inv) * msg_U_f_del_v_inv_m
    self.msg_U_f_m_del[self.num_nodes:,:,T] = self.msg_U_b_m[self.num_nodes:,:,T]
    
    # backward
    msg_U_b_del_v_inv = \
        1.0/self.msg_U_f_v[:self.num_nodes,:,T] + 1.0/self.msg_U_llk_v[:,:,T]
    msg_U_b_del_v_inv_m = \
        torch.div(self.msg_U_f_m[:self.num_nodes,:,T],self.msg_U_f_v[:self.num_nodes,:,T])\
        +torch.div(self.msg_U_llk_m[:,:,T],self.msg_U_llk_v[:,:,T])
        
    self.msg_U_b_v_del[:self.num_nodes,:,T] = 1.0/msg_U_b_del_v_inv
    self.msg_U_b_v_del[self.num_nodes:,:,T] = self.msg_U_f_v[self.num_nodes:,:,T]
    
    self.msg_U_b_m_del[:self.num_nodes,:,T] = (1.0/msg_U_b_del_v_inv) * msg_U_b_del_v_inv_m
    self.msg_U_b_m_del[self.num_nodes:,:,T] = self.msg_U_f_m[self.num_nodes:,:,T]



def post_update_U(self):
    # merge all factor->var messages: U_llk, U_f, U_b
    # we only use it for init/ merge-all after training process

    post_U_v_inv_all = 1.0/self.msg_U_f_v + 1.0/self.msg_U_b_v + 1.0/self.msg_U_llk_v
    post_U_v_inv_m_all = torch.div(self.msg_U_f_m,self.msg_U_f_v)\
                        +torch.div(self.msg_U_b_m,self.msg_U_b_v)\
                         +torch.div(self.msg_U_llk_m,self.msg_U_llk_v)
    
    
    # arrange the post.U per mode
    self.post_U_m = []
    self.post_U_v = []
    
    idx_start = 0     
         
    for ndim in self.ndims:
        idx_end = idx_start + ndim
        post_U_mode_v = 1.0/post_U_v_inv_all[idx_start:idx_end,:,:]
        post_U_mode_m = post_U_mode_v * post_U_v_inv_m_all[idx_start:idx_end,:,:]
        
        self.post_U_v.append(post_U_mode_v)
        self.post_U_m.append(post_U_mode_m)
        
        idx_start = idx_end
                    
    
def post_update_tau(self):
    self.post_a = self.a0 + self.msg_tau_a.sum() - self.N_time
    self.post_b = self.b0 + self.msg_tau_n.sum()  
        
        
        
        
        
        