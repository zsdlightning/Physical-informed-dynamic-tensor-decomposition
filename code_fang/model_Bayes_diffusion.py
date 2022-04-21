import numpy as np
import scipy
# import pandas
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
        self.y_tr = torch.tensor(data_dict['tr_y']).to(self.device) # N*1

        self.ind_te = data_dict['te_ind']
        self.y_te = torch.tensor(data_dict['te_y']).to(self.device) # N*1

        self.N = len(data_dict['tr_y'])

        self.ndims = data_dict['ndims']
        self.nmod = len(self.ndims)
        self.num_nodes = sum(self.ndims)
        
        self.train_time_ind =data_dict['tr_T_disct'] # N*1
        self.test_time_ind = data_dict['te_T_disct'] # N*1
        
        self.time_uni = data_dict['time_uni'] # N_time*1
        self.N_time = len(self.time_uni) 
        
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
        

        
        self.msg_tau_a = torch.ones(self.N_time,1).to(self.device)
        self.msg_tau_b = torch.ones(self.N_time,1).to(self.device)
        
        # actually, it's the massage from transition-factor -> variabel U
        # for here, we arrange the U-msg by concat-all-as-tensor for efficient computing in transition
        # recall, with Matern 23 kernel, msg_U_transition = [ U, U'], so the firsr-dim is 2*num_nodes 
        
        self.msg_U_f_m = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device) 
        self.msg_U_f_v = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        
        self.msg_U_b_m = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        self.msg_U_b_v = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)   


        # set the start and end factor

        for r in range(self.R_U):
            self.msg_U_b_m[:,r,self.N_time-1] = 0
            self.msg_U_b_v[:,r,self.N_time-1] = 1e8

            self.msg_U_f_m[:,r,0] = 0
            self.msg_U_f_v[:,r,0] = torch.diag(self.P_inf)

    
        # init the calibrating factors / q_del in draft, init/update with current msg
        
        # actually, it's the massage from variabel U -> llk-factor
        self.msg_U_llk_m_del = torch.rand(self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        self.msg_U_llk_v_del = torch.ones(self.num_nodes,self.R_U,self.N_time).double().to(self.device)
          
        self.msg_tau_a_del_T = torch.ones(self.N_time,1).to(self.device)
        self.msg_tau_b_del_T = torch.ones(self.N_time,1).to(self.device)
                
        # actually, it's the massage from variabel U -> trans-factor
        self.msg_U_f_m_del = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device) 
        self.msg_U_f_v_del = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        
        self.msg_U_b_m_del = torch.rand(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device) 
        self.msg_U_b_v_del = torch.ones(2*self.num_nodes,self.R_U,self.N_time).double().to(self.device) 
        
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
        

    def msg_update_U_llk(self,T):
        
        # retrive the observed entries at T
        eind_T = self.time_data_table_tr[T] # id of observed entries at this time-stamp
        N_T = len(eind_T) 
        ind_T = self.ind_tr[eind_T]
        y_T = self.y_tr[eind_T].squeeze()
        
        # uid_table, _ = build_id_key_table(self.nmod,ind_T) # get the id of associated nodes
        # self.uid_table_T = uid_table
        # self.ind_T = ind_T
        
        U_llk_del_T = torch.cat([self.msg_U_llk_m_del[:,:,T],self.msg_U_llk_v_del[:,:,T]]) # concat the m and v 
        U_llk_del_T.requires_grad=True 
        
        U_llk_del_T_m,U_llk_del_T_v = self.arrange_U_llk(U_llk_del_T) # arrange U as mode-wise
        
        E_z_del, E_z_2_del = self.moment_product_U_del(ind_T,U_llk_del_T_m,U_llk_del_T_v) # first and second moment of CP-pred

        self.msg_update_tau_del(T)
        E_tau_del = self.msg_tau_a_del_T[T]/self.msg_tau_b_del_T[T]
        
        log_Z = 0.5*N_T*torch.log(E_tau_del/(2*np.pi)) \
            -  0.5*E_tau_del* ( (y_T*y_T).sum() - 2* (y_T*E_z_del).sum() + E_z_2_del.sum())

        log_Z.backward()

        U_llk_del_grad = U_llk_del_T.grad

        U_llk_del_m_grad = U_llk_del_grad[:self.num_nodes]
        U_llk_del_v_grad = U_llk_del_grad[self.num_nodes:]

        # ADF update
        U_llk_m_star = self.msg_U_llk_m_del[:,:,T] + self.msg_U_llk_v_del[:,:,T] * U_llk_del_m_grad
        
        U_llk_v_star = self.msg_U_llk_v_del[:,:,T]\
                        - torch.square(self.msg_U_llk_v_del[:,:,T]) * (torch.square(U_llk_del_m_grad)-2*U_llk_del_v_grad) 

        # msg update U_llk: f_star / f_del
        # set as constant for nan/inf case 
        msg_U_llk_v_inv = 1.0/U_llk_v_star - 1.0/self.msg_U_llk_v_del[:,:,T]
        msg_U_llk_v_inv_m = torch.div(U_llk_m_star,U_llk_v_star) \
                        - torch.div(self.msg_U_llk_m_del[:,:,T],self.msg_U_llk_v_del[:,:,T])

        self.msg_U_llk_v[:,:,T] = 1.0/msg_U_llk_v_inv
        self.msg_U_llk_m[:,:,T] = (1.0/msg_U_llk_v_inv) * msg_U_llk_v_inv_m


        # we also update the tau here
        a = 0.5*N_T + 1
        b = 0.5*((y_T*y_T).sum() - 2* (y_T*E_z_del).sum() + E_z_2_del.sum()).detach()
        self.msg_update_tau(a,b,T)

    def arrange_U_llk(self,U_llk_del_T):
        # arrange_U_to_mode-wise for convenience in computing
        U_llk_del_T_m = U_llk_del_T[:self.num_nodes,:]
        U_llk_del_T_v = U_llk_del_T[self.num_nodes:,:]
        
        # arrange_U_to_mode-wise for convenience in computing
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
            
        return U_llk_del_T_m,U_llk_del_T_v
        
        
    def moment_product_U_del(self,ind_T,U_llk_T_m,U_llk_T_v):
        # double check E_z_2:done
        # compute first and second moments of \Hadmard_prod_{k \in given modes} u_k -CP based on the U_llk_del
        # based on the U_llk_del (calibrating factors)

        E_z = U_llk_T_m[0][ind_T[:,0]] # N*R_u
        E_z_expand = E_z.unsqueeze(-1) # N*R_u*1
        E_z_expand_T = torch.transpose(E_z_expand, dim0=1, dim1=2)# N*1*R_u
        E_z_2 = torch.diag_embed(U_llk_T_v[0][ind_T[:,0]],dim1=1) + torch.bmm(E_z_expand,E_z_expand_T) # N*R_u*R_u

        for mode in range(1,self.nmod):
            E_u = U_llk_T_m[mode][ind_T[:,mode]] # N*R_u
            E_u_expand = E_u.unsqueeze(-1) # N*R_u*1
            E_u_expand_T = torch.transpose(E_u_expand, dim0=1, dim1=2)# N*1*R_u
            E_u_2 = torch.diag_embed(U_llk_T_v[mode][ind_T[:,mode]],dim1=1) + torch.bmm(E_u_expand,E_u_expand_T) # N*R_u*R_u

            E_z = E_z*E_u
            E_z_2 = E_z_2*E_u_2  
        
        # E(1^T z)^2 = trace (1*1^T* z^2)
        if self.R_U>1:
            return E_z.squeeze().sum(-1), torch.einsum('bii->b',\
                                            torch.matmul(E_z_2,torch.ones(self.R_U,self.R_U).double().to(self.device) ))
        else:
            return E_z.squeeze(), torch.einsum('bii->b',E_z_2)
        


    def msg_update_tau(self,a,b,T):
        self.msg_tau_a[T] = a
        self.msg_tau_b[T] = b
        
    def msg_update_tau_del(self,T):
        # add prior db check: done
        self.msg_tau_a_del_T[T] = self.a0 + self.msg_tau_a[:T].sum() + self.msg_tau_a[T+1:].sum() - self.N_time
        self.msg_tau_b_del_T[T] = self.b0 + self.msg_tau_b[:T].sum() + self.msg_tau_b[T+1:].sum()

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
                                
        
    def msg_update_U_trans_del(self,T, mode='forward'):
        # U_f_del = U_b + U_llk : msg from var U_T to p(U_T | U_{T-1}) 
        # (left direction msg, will used during the backward)
        
        # U_b_del = U_f + U_llk : msg from var U_T to p(U_{T+1} | U_{T})  
        # (right direction msg, will used during the forward)
        
        # double check:done

        if mode=='forward' and T<self.N_time-1:
            # for the last time var, we don't have to update its U_b_del (right-out msg)--we'll never use it

            msg_U_b_del_v_inv = \
                1.0/self.msg_U_f_v[:self.num_nodes,:,T] + 1.0/self.msg_U_llk_v[:,:,T]
            
            msg_U_b_del_v_inv_m = \
                torch.div(self.msg_U_f_m[:self.num_nodes,:,T],self.msg_U_f_v[:self.num_nodes,:,T])\
                +torch.div(self.msg_U_llk_m[:,:,T],self.msg_U_llk_v[:,:,T])
                
            self.msg_U_b_v_del[:self.num_nodes,:,T] = 1.0/msg_U_b_del_v_inv
            self.msg_U_b_v_del[self.num_nodes:,:,T] = self.msg_U_f_v[self.num_nodes:,:,T]
            
            self.msg_U_b_m_del[:self.num_nodes,:,T] = (1.0/msg_U_b_del_v_inv) * msg_U_b_del_v_inv_m
            self.msg_U_b_m_del[self.num_nodes:,:,T] = self.msg_U_f_m[self.num_nodes:,:,T]



        else:
            # backward
            if T>0:
            # for the T0 var, we don't update its U_f_del (left-out msg)--we'll never use it

                msg_U_f_del_v_inv = \
                    1.0/self.msg_U_b_v[:self.num_nodes,:,T] + 1.0/self.msg_U_llk_v[:,:,T]
                msg_U_f_del_v_inv_m = \
                    torch.div(self.msg_U_b_m[:self.num_nodes,:,T],self.msg_U_b_v[:self.num_nodes,:,T])\
                    +torch.div(self.msg_U_llk_m[:,:,T],self.msg_U_llk_v[:,:,T])
                    
                self.msg_U_f_v_del[:self.num_nodes,:,T] = 1.0/msg_U_f_del_v_inv
                self.msg_U_f_v_del[self.num_nodes:,:,T] = self.msg_U_b_v[self.num_nodes:,:,T]
                
                self.msg_U_f_m_del[:self.num_nodes,:,T] = (1.0/msg_U_f_del_v_inv) * msg_U_f_del_v_inv_m
                self.msg_U_f_m_del[self.num_nodes:,:,T] = self.msg_U_b_m[self.num_nodes:,:,T]
            




    def post_update_U(self):
        # merge all factor->var messages: U_llk, U_f, U_b
        # we only use it for init/ merge-all after training process

        post_U_v_inv_all = 1.0/self.msg_U_f_v[:self.num_nodes,:,:] + 1.0/self.msg_U_b_v[:self.num_nodes,:,:] + 1.0/self.msg_U_llk_v
        post_U_v_inv_m_all = torch.div(self.msg_U_f_m[:self.num_nodes,:,:],self.msg_U_f_v[:self.num_nodes,:,:])\
                            +torch.div(self.msg_U_b_m[:self.num_nodes,:,:],self.msg_U_b_v[:self.num_nodes,:,:])\
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
            
            
    def msg_update_U_trans(self,T,mode='forward'):
        
        time_gap = self.time_uni[T+1] - self.time_uni[T]
        A_T = torch.matrix_exp(self.F * time_gap)
        Q_T = self.P_inf - self.P_inf @ A_T @ self.P_inf.T

        for r in range(self.R_U):
            # msg from the left (from U_T)
            # double check: f/b:done

            msg_m_l = self.msg_U_b_m_del[:,r,T]
            msg_v_l = self.msg_U_b_v_del[:,r,T]
            
            # msg from the right (from U_{T+1})
            msg_m_r = self.msg_U_f_m_del[:,r,T+1]
            msg_v_r = self.msg_U_f_v_del[:,r,T+1]

            if mode=='forward':
            #  in the forward pass, we only update the msg to right (U_{T+1})
                msg_m_r.requires_grad=True 
                msg_v_r.requires_grad=True 
                target_m = msg_m_r
                target_v = msg_v_r
            else:
            # in the backward pass, we only update the msg to left (U_{T})
                msg_m_l.requires_grad=True 
                msg_v_l.requires_grad=True 
                target_m = msg_m_l
                target_v = msg_v_l            

            mu = (A_T @ msg_m_l.view(-1,1)).squeeze() # num_node * 1
            sigma = torch.diag(msg_v_r) + Q_T + A_T @ torch.diag(msg_v_l) @ A_T.T
            sample = msg_m_r

            # compute log-Z
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
            log_Z_trans = dist.log_prob(sample)
            log_Z_trans.backward()

            # get grad and ADF update 
            target_m_grad = target_m.grad
            target_v_grad = target_v.grad

            target_m = target_m.detach()
            target_v = target_v.detach()

            target_m_star = target_m + target_v * target_m_grad
            target_v_star = target_v - torch.square(target_v) * (torch.square(target_m_grad) - 2*target_v_grad)

            # update the factor: msg_new = msg_star / msg_old
            target_v_inv_new = 1.0/target_v_star - 1.0/target_v
            target_v_inv_m_new = torch.div(target_m_star,target_v_star) -torch.div(target_m,target_v)


            # DOUBLE CHECK:done
            if mode=='forward':
                self.msg_U_f_v[:,r,T+1] = 1.0/target_v_inv_new
                self.msg_U_f_m[:,r,T+1] = (1.0/target_v_inv_new) * target_v_inv_m_new
            else:
                self.msg_U_b_v[:,r,T] = 1.0/target_v_inv_new
                self.msg_U_b_m[:,r,T] = (1.0/target_v_inv_new) * target_v_inv_m_new

    def msg_update_U_trans_vec(self,T,mode='forward'):
        
        time_gap = self.time_uni[T+1] - self.time_uni[T]
        A_T_block = torch.block_diag(*([torch.matrix_exp(self.F * time_gap)]*self.R_U))
        P_inf_block = torch.block_diag(*([self.P_inf]*self.R_U))
        Q_T_block = P_inf_block - P_inf_block @ A_T_block @ P_inf_block.T

        msg_m_l = self.msg_U_f_m_del[:,:,T].T.reshape(-1)
        msg_v_l = self.msg_U_f_v_del[:,:,T].T.reshape(-1)
            
        # msg from the right (from U_{T+1})
        msg_m_r = self.msg_U_b_m_del[:,:,T+1].T.reshape(-1)
        msg_v_r = self.msg_U_b_v_del[:,:,T+1].T.reshape(-1)

        if mode=='forward':
        #  in the forward pass, we only update the msg to right (U_{T+1})
            msg_m_r.requires_grad=True 
            msg_v_r.requires_grad=True 
            target_m = msg_m_r
            target_v = msg_v_r
        else:
        # in the backward pass, we only update the msg to left (U_{T})
            msg_m_l.requires_grad=True 
            msg_v_l.requires_grad=True 
            target_m = msg_m_l
            target_v = msg_v_l 

        mu = (A_T_block @ msg_m_l.view(-1,1)).squeeze() # num_node * 1
        sigma = torch.diag(msg_v_r) + Q_T_block + A_T_block @ torch.diag(msg_v_l) @ A_T_block.T
        sample = msg_m_r

        # print(sample)

        # compute log-Z
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
        log_Z_trans = dist.log_prob(sample)
        log_Z_trans.backward()

        # get grad and ADF update 
        target_m_grad = target_m.grad
        target_v_grad = target_v.grad

        target_m = target_m.detach()
        target_v = target_v.detach()

        target_m_star = target_m + target_v * target_m_grad
        target_v_star = target_v - torch.square(target_v) * (torch.square(target_m_grad) - 2*target_v_grad)

        # update the factor: msg_new = msg_star / msg_old
        target_v_inv_new = 1.0/target_v_star - 1.0/target_v
        target_v_inv_m_new = torch.div(target_m_star,target_v_star) -torch.div(target_m,target_v)


        if mode=='forward':
            self.msg_U_b_v_del[:,:,T+1] = (1.0/target_v_inv_new).reshape(self.R_U,2*self.num_nodes).T
            self.msg_U_b_m_del[:,:,T+1] = ((1.0/target_v_inv_new) * target_v_inv_m_new).reshape(self.R_U,2*self.num_nodes).T
        else:
            self.msg_U_f_v_del[:,:,T] =  (1.0/target_v_inv_new).reshape(self.R_U,2*self.num_nodes).T
            self.msg_U_f_m_del[:,:,T] = ((1.0/target_v_inv_new) * target_v_inv_m_new).reshape(self.R_U,2*self.num_nodes).T



    def model_test(self):


        pass


        
        
        