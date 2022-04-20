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
        
        self.ind_tr = data_dict['ind_tr']
        self.y_tr = data_dict['y_tr'].to(self.device) # N*1
        self.N = len(data_dict['y_tr'])
        self.ndims = data_dict['ndims']
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
        
        # uid-data table 
        self.uid_table, self.data_table = build_id_key_table(self.nmod,self.ind_tr) 
        

def msg_update_U_llk(self,T):
    pass
    
def msg_update_U_trans(self,T,mode='forward'):
    pass

def msg_update_tau(self,T):
    pass

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
    self.msg_U_llk_m_del[:,:,T] = (1.0/msg_U_llk_del_v_inv) @ msg_U_llk_del_v_inv_m
                            
    
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
    
    self.msg_U_f_m_del[:self.num_nodes,:,T] = (1.0/msg_U_f_del_v_inv) @ msg_U_f_del_v_inv_m
    self.msg_U_f_m_del[self.num_nodes:,:,T] = self.msg_U_b_m[self.num_nodes:,:,T]
    
    # backward
    msg_U_b_del_v_inv = \
        1.0/self.msg_U_f_v[:self.num_nodes,:,T] + 1.0/self.msg_U_llk_v[:,:,T]
    msg_U_b_del_v_inv_m = \
        torch.div(self.msg_U_f_m[:self.num_nodes,:,T],self.msg_U_f_v[:self.num_nodes,:,T])\
        +torch.div(self.msg_U_llk_m[:,:,T],self.msg_U_llk_v[:,:,T])
        
    self.msg_U_b_v_del[:self.num_nodes,:,T] = 1.0/msg_U_b_del_v_inv
    self.msg_U_b_v_del[self.num_nodes:,:,T] = self.msg_U_f_v[self.num_nodes:,:,T]
    
    self.msg_U_b_m_del[:self.num_nodes,:,T] = (1.0/msg_U_b_del_v_inv) @ msg_U_b_del_v_inv_m
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
        post_U_mode_m = post_U_mode_v @ post_U_v_inv_m_all[idx_start:idx_end,:,:]
        
        self.post_U_v.append(post_U_mode_v)
        self.post_U_m.append(post_U_mode_m)
        
        idx_start = idx_end
                    
    
def post_update_tau(self):
    self.post_a = self.a0 + self.msg_tau_a.sum() - self.N_time
    self.post_b = self.b0 + self.msg_tau_n.sum()  
        
        
        
        
        
        