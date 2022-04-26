import numpy as np
import scipy
# import pandas
import torch
import utils
from utils import generate_state_space_Matern_23
from scipy import linalg
from utils import build_id_key_table


# discretize the time-stamp as extra-mode, and update it through ADF
# try different ways to feed training data: one by one / group by T-mode / mini-batch

class static_ADF():
    def __init__(self,data_dict,hyper_dict):

        self.R_U = hyper_dict['R_U'] # rank of latent factor of embedding
        self.device = hyper_dict['device']
        self.DAMPING = hyper_dict['DAMPING']
        
        self.a0 = hyper_dict['a0']
        self.b0 = hyper_dict['b0']

        self.m0 = torch.tensor(1.0)
        self.v0 = torch.tensor(1e0)
        
        # data-dependent paras
        self.data_dict = data_dict
        
        self.ind_tr = data_dict['tr_ind']
        self.y_tr = torch.tensor(data_dict['tr_y']).to(self.device) # N*1

        self.ind_te = data_dict['te_ind']
        self.y_te = torch.tensor(data_dict['te_y']).to(self.device) # N*1

        self.N = len(data_dict['tr_y'])

        self.ndims = data_dict['ndims']

        # print(self.ndims)

        self.nmod = len(self.ndims)
        self.num_nodes = sum(self.ndims)
        
        # self.train_time_ind =data_dict['tr_T_disct'] # N*1
        # self.test_time_ind = data_dict['te_T_disct'] # N*1
        
        # self.time_uni = data_dict['time_uni'] # N_time*1
        # self.N_time = len(self.time_uni) 

        self.post_U_m = self.m0*torch.rand(self.num_nodes,self.R_U).double().to(self.device)
        self.post_U_v = self.v0*torch.ones(self.num_nodes,self.R_U).double().to(self.device)

        # print(self.ind_tr.shape)
        # self.msg_U_v_inv_m = self.m0*torch.rand(self.num_nodes,self.R_U,self.N_time).double().to(self.device)
        # self.msg_U_v_inv = self.v0*torch.ones(self.num_nodes,self.R_U,self.N_time).double().to(self.device)

        # self.tau_a_T = torch.ones(self.N_time,1).to(self.device)
        # self.tau_b_T = torch.ones(self.N_time,1).to(self.device)

        self.tau_a_N = torch.ones(self.N,1).to(self.device)
        self.tau_b_N = torch.ones(self.N,1).to(self.device)

        self.tau=0.1

        # self.time_data_table_tr = utils.build_time_data_table(self.train_time_ind) 
        # self.time_data_table_te = utils.build_time_data_table(self.test_time_ind) 


    def CEP_update_T(self,T):

        eind_T = self.time_data_table_tr[T] # id of observed entries at this time-stamp
        N_T = len(eind_T) 
        ind_T = self.ind_tr[eind_T]

        y_T = self.y_tr[eind_T].squeeze()



        tau_a_del_T = self.a0 + self.tau_a_T[:T].sum() + self.tau_a_T[T+1:].sum() - self.N_time +1
        tau_b_del_T = self.b0 + self.tau_b_T[:T].sum() + self.tau_b_T[T+1:].sum()

        E_tau = tau_a_del_T/tau_b_del_T

        uid_table, data_table = build_id_key_table(self.nmod,ind_T)

        post_U_v_inv_T = 1.0/self.msg_U_v_inv[:,:,:T].sum(-1) + 1.0/self.msg_U_v_inv[:,:,T+1].sum(-1)
        post_U_v_inv_m_T = 1.0/self.msg_U_v_inv_m[:,:,:T].sum(-1) + 1.0/self.msg_U_v_inv_m[:,:,T+1].sum(-1)
        
        post_U_v_T = 1.0/post_U_v_inv_T
        post_U_m_T = torch.div(post_U_v_inv_m_T,post_U_v_inv_T)

        U_llk = torch.cat([post_U_m_T,post_U_v_T]) # concat the m and v and set grad
        
        post_U_m,post_U_v = self.arrange_U_llk(U_llk)

        for mode in range(self.nmod):

            E_z_del,E_z_del_2 = self.expectation_update_z_del(ind_T,del_mode,post_U_m,post_U_v)
            
            for j in range(len(uid_table[mode])):

                uid = uid_table[mode][j] # id of embedding
                eid = data_table[mode][j]

                # compute msg of associated entries (but we don't store them)
                msg_U_v_inv = E_tau * E_z_del_2[eid]  # num_eid * R_U * R_U
                msg_U_m = torch.linalg.solve(msg_U_v_inv, E_tau * torch.bmm(self.E_z_del[eid], self.y_tr[eid].unsqueeze(-1))) # num_eid * R_U *1
        
    def expectation_update_z_del(self,del_mode,ind_T):
        # compute E_z_del,E_z_del_2 by current post.U and deleting the info of mode_k
        # only use in training step
        
        other_modes = [i for i in range(self.nmod)]
        other_modes.remove(del_mode)
        
        init_mode = other_modes[0]
        
        self.E_z_del = post_U_m[init_mode][ind_T[:,init_mode]] # N*R_u*1
        
        E_z_del_T = torch.transpose(self.E_z_del, dim0=1, dim1=2)# N*1*R_u
        
        self.E_z_del_2 = self.post_U_v[init_mode][self.ind_tr[:,init_mode]] + torch.bmm(self.E_z_del,E_z_del_T) # N*R_u*R_u
        
        for mode in other_modes[1:]:
            
            cur_z = self.post_U_m[mode][self.ind_tr[:,mode]]# N*R_u*1
            cur_z_T =  torch.transpose(cur_z, dim0=1, dim1=2)# N*1*R_u
            cur_z_2 = self.post_U_v[mode][self.ind_tr[:,mode]] + torch.bmm(cur_z,cur_z_T)
            
            self.E_z_del = self.E_z_del * cur_z
            self.E_z_del_2 = self.E_z_del_2 * cur_z_2 # N*R_u*R_u
             




    def ADF_update_T(self,T):
        # training data feed in with each group of T
        
        eind_T = self.time_data_table_tr[T] # id of observed entries at this time-stamp
        N_T = len(eind_T) 
        ind_T = self.ind_tr[eind_T]

        y_T = self.y_tr[eind_T].squeeze()

        U_llk = torch.cat([self.post_U_m,self.post_U_v]) # concat the m and v and set grad
        U_llk.requires_grad=True 

        U_llk_m,U_llk_v = self.arrange_U_llk(U_llk) # arrange U as mode-wise

        E_z_del, E_z_2_del = self.moment_product_U_del(ind_T,U_llk_m,U_llk_v) # first and second moment of CP-pred


        tau_a_del_T = self.a0 + self.tau_a_T[:T].sum() + self.tau_a_T[T+1:].sum() - self.N_time +1
        tau_b_del_T = self.b0 + self.tau_b_T[:T].sum() + self.tau_b_T[T+1:].sum()
        

        E_tau_del = tau_a_del_T/tau_b_del_T
        # print(E_tau_del)

        log_Z = 0.5*N_T*torch.log(E_tau_del/(2*np.pi)) \
            -  0.5*E_tau_del* ( (y_T*y_T).sum() - 2* (y_T*E_z_del).sum() + E_z_2_del.sum())

        log_Z.backward()

        # mu = E_z_del
        # sigma = (1.0/E_tau_del) * torch.eye(N_T).double()+torch.diag(E_z_2_del)
        # sample =  y_T
        # dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
        # log_Z_trans = dist.log_prob(sample)

        # log_Z_trans.backward()

        U_llk_grad = U_llk.grad
        U_llk_m_grad = U_llk_grad[:self.num_nodes]
        U_llk_v_grad = U_llk_grad[self.num_nodes:]
        

        # ADF update
        U_llk_m_star = self.post_U_m + self.post_U_v * U_llk_m_grad
        
        U_llk_v_star = self.post_U_v\
                        - torch.square(self.post_U_v) * \
                            (torch.square(U_llk_m_grad)-2*U_llk_v_grad) 

        # DAMPING
        U_llk_v_inv = self.DAMPING * (1.0/self.post_U_v) + (1-self.DAMPING) * (1.0/U_llk_v_star)
        U_llk_v_inv_m = self.DAMPING * torch.div(self.post_U_m,self.post_U_v) + (1-self.DAMPING) * torch.div(U_llk_m_star,U_llk_v_star)


        U_llk_v_inv = torch.where(U_llk_v_star>0,U_llk_v_star,1.0)

        self.post_U_m = torch.nan_to_num(torch.div(U_llk_v_inv_m,U_llk_v_inv))
        self.post_U_v = torch.nan_to_num(1.0/U_llk_v_inv)

        a = 0.5*N_T + 1
        b = 0.5*((y_T*y_T).sum() - 2* (y_T*E_z_del).sum() + E_z_2_del.sum()).detach()
        self.msg_update_tau(a,b,T)

        
    def ADF_update_N(self,n):
        # entry-wise update embedding
        
        ind = self.ind_tr[n]
        y = self.y_tr[n]
        # print(ind)
        embed_m = []
        embed_v = []


        start_idx = 0
        for mode, dim in enumerate(self.ndims):

            U_m = self.post_U_m[start_idx+ind[mode]].reshape(-1,1)
            U_m.requires_grad = True
            

            U_v = self.post_U_v[start_idx+ind[mode]].reshape(-1,1)
            U_v.requires_grad = True

            embed_m.append(U_m)
            embed_v.append(U_v)

            start_idx = start_idx + dim

            # print(start_idx)

        E_z = embed_m[0]
        E_z_2 = torch.diag(embed_v[0]) + torch.mm(E_z,E_z.T)

        
        for mode in range(1,self.nmod):
            E_u = embed_m[mode]
            E_u_2 = torch.diag(embed_v[mode]) + torch.mm(E_u,E_u.T)

            E_z = E_z*E_u
            E_z_2 = E_z_2*E_u_2  
        
        E_z = E_z.sum()
        E_z_2 = E_z_2.sum()

        tau_a_del_n = self.a0 + self.tau_a_N[:n].sum() + self.tau_a_N[n+1:].sum() - self.N +1
        tau_b_del_n = self.b0 + self.tau_b_N[:n].sum() + self.tau_b_N[n+1:].sum()
        

        E_tau_del = tau_a_del_n/tau_b_del_n
        # print(E_tau_del)

        # log_Z = 0.5*torch.log(E_tau_del/(2*np.pi)) \
        #     -  0.5*E_tau_del* ( (y*y) - 2* (y*E_z) + E_z_2)
        # log_Z.backward()

        mu = E_z
        sigma = torch.sqrt((1.0/E_tau_del) + E_z_2)
        sample =  y
        dist = torch.distributions.normal.Normal(mu, sigma)
        log_Z = dist.log_prob(sample)
        log_Z.backward()

        

        start_idx = 0
        for mode, dim in enumerate(self.ndims):
            grad_m = embed_m[mode].grad
            grad_v = embed_v[mode].grad
            # torch.autograd.set_detect_anomaly(True)
            # grad_m = torch.autograd.grad(log_Z,embed_m[mode],retain_graph = True)[0]#embed_m[mode].grad
            # grad_v = torch.autograd.grad(log_Z,embed_v[mode],retain_graph = True)[0]#embed_v[mode].grad

            m_star = embed_m[mode] + embed_v[mode] * grad_m
            v_star = embed_v[mode]\
                        - torch.square(embed_v[mode]) * \
                            (torch.square(grad_m)-2*grad_v) 

            self.post_U_m[start_idx+ind[mode],:] = m_star.detach().squeeze()
            self.post_U_v[start_idx+ind[mode],:] = v_star.detach().squeeze()


        a = 0.5 + 1
        b = 0.5*((y*y) - 2* (y*E_z) + E_z_2).detach()
        self.tau_a_N[n] = a
        self.tau_b_N[n] = b





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

    def grad_zero(self,paras):
        for item in paras:
            if item.grad is not None:
                item.grad.detatch_()
                item.grad.zero_()

    def msg_update_tau(self,a,b,T):
        self.tau_a_T[T] = a
        self.tau_b_T[T] = b

    def model_test(self):
        # y_pred_list = []



        U_llk = torch.cat([self.post_U_m,self.post_U_v]) # concat the m and v and set grad
        U_llk.requires_grad=True 

        U_llk_m,U_llk_v = self.arrange_U_llk(U_llk) # arrange U as mode-wise
        ind_T = self.ind_te

        E_z = U_llk_m[0][ind_T[:,0]] # N*R_u

        for mode in range(1,self.nmod):
            E_u = U_llk_m[mode][ind_T[:,mode]] # N*R_u

            E_z = E_z*E_u
                
            # print(E_z.sum(-1).shape)
        
        
        y_pred=(E_z.sum(-1).squeeze())

        loss = torch.nn.MSELoss()
        rmse = torch.sqrt(loss(y_pred,self.y_te.squeeze()))

        return rmse

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