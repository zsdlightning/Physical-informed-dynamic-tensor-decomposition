import numpy as np
from numpy.lib import utils
import torch 
import matplotlib.pyplot as plt
from model import LDS_GP
# from model_CP import LDS_CEP_full_v3
import os
import tqdm
import utils
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
JITTER = 1e-4

torch.manual_seed(2)

# tucker-form dynamic tensor factorization: decompose form

# gamma is coorespoding to the tucker-core decompose: W_1,W_2 .. W_K in draft
# z is coorespoding to alpha =  \prod W_k*u_k in draft
# z_del  is coorespoding to alpha^{\k} = \prod_{j \neq k} W_j*u_j in draft

# for each W_k, build a LDS for inference
# multi-llk version

class LDS_dynamic_CP():
    def __init__(self,hyper_para_dict):

        super(dynamic_CP,self).__init__(hyper_para_dict)
        
        # for each mode, we assign 
        self.msg_gamma_lam = [1e-4*torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*R_U)* nmod
        self.msg_gamma_eta = [torch.zeros(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)]# (N*R_U*1) * nmod

        # will be update based on nature paras msg
        self.msg_gamma_m = [torch.zeros(self.N_time,self.R_U,1).double().to(self.device)  for i in range(self.nmod)] #  (N*R_U*1) * nmod
        self.msg_gamma_v = [torch.zeros(self.N_time,self.R_U,self.R_U).double().to(self.device)  for i in range(self.nmod) for i in range(self.nmod)] # (N*R_U*R_U) * nmod 

        # value will be updated after filter&smooth
        self.post_gamma_m = [torch.zeros(self.N_time,self.R_U,1).double().to(self.device) for i in range(self.nmod)] #  (N*R_U*1) * nmod
        self.post_gamma_v = [torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N_time,1,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*R_U) * nmod

        # self.E_gamma = [torch.ones(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)]
        self.E_gamma = [torch.ones(self.N_time,self.R_U,1).double().to(self.device) for i in range(self.nmod)]
        self.E_gamma_2 = [torch.ones(self.N_time,self.R_U,self.R_U).double().to(self.device) for i in range(self.nmod)]
        
        # coorespoding to the z_backslash in draft
        self.E_z_del = [None for i in range(self.nmod)] 
        self.E_z_del_2 = [None for i in range(self.nmod)] 

        for mode in range(self.nmod):
            self.expectation_update_z_del(mode)

        self.expectation_update_z(self.ind_tr)

        # some constant terms 
        self.ones_const = torch.ones(self.N,1).to(self.device)

    def moment_prod_tucker(self,modes,ind):
        # computhe first and second moments of prod_{k \in given modes} W_k * u_k (denote as z/z^{\k} or alpha/alpha^{\k} in draft)
        N = ind.shape[0]
        E_z = torch.ones(N,1,1).to(self.device)
        E_z_2 = torch.ones(N,1,1).to(self.device)

        for mode in modes:

            E_u = self.post_U_m[mode][ind[:,mode]] # N*R_u*1
            E_u_2 = self.post_U_v[mode][ind[:,mode]] + torch.bmm(E_u,E_u.transpose(dim0=1,dim1=2)) # N*R_u*R_U
            
            E_W = self.E_gamma[mode][self.train_time_ind] # N*R_u*1
            E_W_2 = self.E_gamma_2[mode][self.train_time_ind] # N*R_u*R_U
            
            E_z = E_z * torch.bmm(E_W.transpose(dim0=1,dim1=2),E_u) # N*1*1

            # E_z_2 = E_z_2 * torch.square(torch.bmm(E_W.transpose(dim0=1,dim1=2),E_u))

            E_z_2 = E_z_2 * torch.einsum('bii->b',torch.bmm(E_u_2,E_W_2)).reshape(self.N,1,1)# N*1*1 -two-order estimate

            # E_z_2 = E_z_2 * torch.bmm(E_W.transpose(dim0=1,dim1=2),torch.bmm(E_u_2 ,E_W))

        return E_z,E_z_2
    
    def expectation_update_z_del(self,del_mode):
        # compute E_z_del,E_z_del_2 by current post.U and deleting the info of mode_k
        
        other_modes = [i for i in range(self.nmod)]
        other_modes.remove(del_mode)
        self.E_z_del[del_mode],self.E_z_del_2[del_mode] = self.moment_prod_tucker(other_modes,self.ind_tr)
             
    def expectation_update_z(self,ind): 
        # compute E_z,E_z_2 for given datapoints by current post.U (merge info of all modes )

        all_modes = [i for i in range(self.nmod)]        
        self.E_z,self.E_z_2 = self.moment_prod_tucker(all_modes,ind)  

    def expectation_update_gamma(self):
    
        for i in range(self.nmod):
            self.E_gamma[i] = self.post_gamma_m[i] # N*R_U*1
            self.E_gamma_2[i] = self.post_gamma_v[i] + torch.bmm(self.E_gamma[i],self.E_gamma[i].transpose(dim0=1, dim1=2)) # N*R_U*R_U
    
    def msg_update_tau(self):
        self.msg_a = 1.5*self.ones_const
         
        term1 = 0.5 * torch.square(self.y_tr) # N*1
        term2 = self.y_tr * self.E_z.squeeze(-1) # N*1
        term3 = 0.5 * self.E_z_2.squeeze(-1) # N*1
        
        self.msg_b =  term1 - term2 + term3 # N*1

    def msg_update_gamma(self):
        
        for mode in range(self.nmod):

            E_u = self.post_U_m[mode][self.ind_tr[:,mode]] # N*R_u*1
            E_u_2 = self.post_U_v[mode][self.ind_tr[:,mode]] + torch.bmm(E_u,E_u.transpose(dim0=1,dim1=2)) # N*R_u*R_U

            msg_gamma_lam_new = self.E_tau*self.E_z_del_2[mode]*E_u_2 # N*R_U*R_U
            msg_gamma_eta_new = torch.bmm(E_u,self.y_tr.unsqueeze(-1)) * self.E_tau * self.E_z_del[mode] # N*R_U*1

            self.msg_gamma_lam[mode] = self.DAMPPING_gamma * self.msg_gamma_lam[mode] + (1-self.DAMPPING_gamma)*msg_gamma_lam_new # N*R_U*R_U
            self.msg_gamma_eta[mode] = self.DAMPPING_gamma * self.msg_gamma_eta[mode] + (1-self.DAMPPING_gamma)*msg_gamma_eta_new

            for i in range(self.N_time):
                data_id = self.time_data_table[i]
                self.msg_gamma_v[mode][i] = torch.linalg.inv(self.msg_gamma_lam[mode][data_id].sum(dim=0))  
                self.msg_gamma_m[mode][i] = torch.mm(self.msg_gamma_v[mode][i],self.msg_gamma_eta[mode][data_id].sum(dim=0))# N*R_U*1


    def msg_update_U(self):
        # same with father class
        for mode in range(self.nmod):
            
            self.expectation_update_z_del(mode)
            self.expectation_update_z(self.ind_tr)

            for j in range(len(self.uid_table[mode])):

                uid = self.uid_table[mode][j] # id of embedding
                eid = self.data_table[mode][j] # id of associated entries
                tid = self.train_time_ind[eid] # id time states of such entries
                
                # compute msg of associated entries, update with damping
                msg_U_lam_new = self.E_tau * self.E_z_del_2[mode][eid] * self.E_gamma_2[mode][tid] # num_eid * R_U * R

                msg_U_eta_new =  self.E_tau * torch.bmm(self.E_z_del[mode][eid], self.y_tr[eid].unsqueeze(-1))  * self.E_gamma[mode][tid] # num_eid * R_U *1
                
                self.msg_U_lam[mode][eid] = self.DAMPPING_U * self.msg_U_lam[mode][eid] + (1- self.DAMPPING_U ) * msg_U_lam_new # num_eid * R_U * R_U
                self.msg_U_eta[mode][eid] = self.DAMPPING_U * self.msg_U_eta[mode][eid] + (1- self.DAMPPING_U ) * msg_U_eta_new # num_eid * R_U * 1

    def post_update_gamma(self,mode):

        # update post. factor of gamma based on latest results from LDS system (RTS-smoother)

        smooth_m =  torch.matmul(self.H,torch.cat(self.m_smooth_list,dim=1)).T.unsqueeze(dim=-1) # # N*R_U*1

        # tricky point: batch mat-mul to compute H*P*H^T, where P size: N*2R_U*2R_U, H size: (nmod*R_U)*(nmod*2R_U) 
        P = torch.stack(self.P_smooth_list,dim=0)# N*(2*R_U*nmod)*(2*R_U*nmod)
        # step1: A=P*H.T
        tensor1 = torch.matmul(P,self.H.T) # N* (2R_U) * (R_U)
        # step2: B= H*A = H*P*H.T = (A^T * H^T )^T
        smooth_v = torch.matmul(torch.transpose(tensor1,dim0=1,dim1=2), self.H.T).transpose(dim0=1,dim1=2)# # N*R_U*R_U

        self.post_gamma_m[mode] = smooth_m # N*R_U*1
        self.post_gamma_v[mode] = smooth_v # N*R_U*R_U

    def expectation_update_gamma(self,mode):
    
        self.E_gamma[mode] = self.post_gamma_m[mode] # N*R_U*1
        self.E_gamma_2[mode] = self.post_gamma_v[mode] + torch.bmm(self.E_gamma[mode],self.E_gamma[mode].transpose(dim0=1, dim1=2)) # N*R_U*R_U


    def model_test(self,test_ind,test_y,test_time):
            
            MSE_loss = torch.nn.MSELoss()
            # smooth_result = torch.cat(self.m_smooth_list,dim=1) # size: (2 nmod*R_U)*N

            # train_loss

            # all_modes = [i for i in range(self.nmod)]        
            # E_z_train,_ = self.moment_prod_tucker(all_modes,self.ind_tr)    

            y_pred_train = self.E_z.squeeze()
            # y_pred_train = E_z_train.squeeze()
            y_true_train = self.y_tr.squeeze()
            loss_train =  torch.sqrt(MSE_loss(y_pred_train,y_true_train))

            # test_loss

            N_test = test_ind.shape[0]
            
            E_z_base = torch.ones(N_test,1,1).double().to(self.device)

            tid = self.test_time_ind

            for mode in range(self.nmod):

                E_W = self.E_gamma[mode][self.test_X_state.squeeze()][tid]# N*R_u*1

                E_u = self.post_U_m[mode][test_ind[:,mode]] # N*R_u*1

                # print(E_W.shape,E_u.shape)

                E_z_base = E_z_base * torch.bmm(E_W.transpose(dim0=1,dim1=2),E_u) # N*1*1

            y_pred_test_base = E_z_base.squeeze()
            loss_test_base =  torch.sqrt(MSE_loss(y_pred_test_base,test_y.squeeze().to(self.device)))
                
            return loss_train,loss_test_base