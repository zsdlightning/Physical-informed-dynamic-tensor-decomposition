'''
    File name: model_static_CP_Tucker.py
    Author: Shikai/Xuangu Fang
    Date created: 12/29/2021
    Locate: Shanghai,CN & SLC,US
    Python Version: 3
'''
'''
standard full-tucker, no scalable
'''


import numpy as np
from numpy.core.fromnumeric import transpose
import torch 
# import matplotlib.pyplot as plt
import tensorly as tl
from utils import kronecker_product_einsum_batched,Hadamard_product_batch
tl.set_backend('pytorch')

class static_CEP_base():
    '''
    base class for standard CEP update of CP/full-tucker model
    gamma: here refers lambda(weight vector) in CP or vec(W)(core-tensor) in Tucker
    z or z\: here refers \hadamard_product U_k in CP or \kronecker_product U_k
    U,tau: embedding & inverse-noise
    '''
    
    def __init__(self,hyper_para_dict):
        
        self.device = hyper_para_dict['device'] # add the cuda version later 
        self.N = hyper_para_dict['N'] # number of data-llk

        self.DAMPPING_U = hyper_para_dict['DAMPPING_U']
        self.DAMPPING_gamma = hyper_para_dict['DAMPPING_gamma']

        # training data
        self.ind_tr = hyper_para_dict['ind_tr']
        self.y_tr = hyper_para_dict['y_tr'].to(self.device) # N*1
        
        # some hyper-paras
        # self.epoch = hyper_para_dict['epoch'] # passing epoch
        self.ndims = hyper_para_dict['ndims'] 
        self.U = [item.to(self.device) for item in hyper_para_dict['U']] # list of mode embedding, fixed and known in this setting
        self.nmod = len(self.U)
        self.R_U = hyper_para_dict['R_U'] # rank of latent factor of embedding

        self.nmod_list = [self.R_U for k in range(self.nmod)]
        self.gamma_size = hyper_para_dict['gamma_size'] # R_U for CP, (R_U)^K for tucker
        # prior of noise
        self.v = hyper_para_dict['v'] # prior varience of embedding (scaler)
        self.a0 = hyper_para_dict['a0']
        self.b0 = hyper_para_dict['b0']

        
        # init the message/llk factor

        # nature paras of msg_U and msg_gamma, lam = v_{-1}, eta =  v_{-1}*m

        self.msg_U_lam = [1e-4*torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double().to(self.device) for i in range(self.nmod) ] # (N*R_U*R_U)*nmod
        self.msg_U_eta =  [torch.zeros(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*1)*nmod

        self.msg_gamma_lam = 1e-4*torch.eye(self.gamma_size).reshape((1,self.gamma_size,self.gamma_size)).repeat(self.N,1,1).double().to(self.device) # N*(R^K)*(R^K)
        self.msg_gamma_eta = torch.zeros(self.N,self.gamma_size,1).double().to(self.device) # N*(R^K)*1
        
        # msg of tau
        self.msg_a = torch.ones(self.N,1).double().to(self.device) # N*1
        self.msg_b = torch.ones(self.N,1).double().to(self.device) # N*1


        # init the approx. post factor (over all data points)        
        
        # post. of tau
        self.post_a = self.a0
        self.post_b = self.b0

        # post. of gamma
        self.post_gamma_m = torch.zeros(self.gamma_size,1).double().to(self.device) # (R^K)*1
        self.post_gamma_v = torch.eye(self.gamma_size).double().to(self.device) # (R^K)*(R^K)

        # post.of U     
        self.post_U_m = [item.unsqueeze(-1) for item in self.U] # nmod(list) * (ndim_i * R_U *1) 
        self.post_U_v = [(self.v) *torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(ndim,1,1).double().to(self.device) \
            for ndim in self.ndims]# nmod(list) * (ndim_i * R_U *R_U ) 


        # Expectation terms over current approx. post
                
        self.E_tau = 1.0

        self.E_gamma = None # (R^K)*1
        self.E_gamma_2 = None # (R^K)*(R^K) -> actually, no need to use this

        # we do not store the expecation of each U, but compute & store the z, z_del
        # no need to store the conditional moment for each mode(just compute, use and drop)
        # if we won;t, use list here 
        self.E_z_del = None  # N* (R^{K-1}) *1 in tucker
        self.E_z_del_2 = None # N* (R^{K-1}) * (R^{K-1}) in tucker

        self.E_z = None # N* (R^K) *1 in tucker
        self.E_z_2 = None # N* (R^K) * (R^K)in tucker 

        # uid-data table 
        self.uid_table, self.data_table = self.build_id_key_table() 

        # some constant terms 
        self.ones_const = torch.ones(self.N,1).to(self.device)
        self.eye_const = torch.eye(self.R_U).to(self.device)

        # batch-product function: 
        # assgain as \kronecker_product in Tucker, \Hadamard_product
        self.batch_product = None

        # compute the E_gamma form when delete info of given mode
        # For CP: E_gamma_del_mode = E_gamma
        # For Tucker:  E_gamma_del_mode = unfold(E_gamma).fold(mode),
        #              unfolding the gamma/W (vec->tensor->matrix):
        self.E_gamma_del_mode_func = None

    def build_id_key_table(self):
        # build uid-data_key_table, implement by nested list
        
        # given indices of unique rows of each mode/embed (store in uid_table)  
        uid_table = []
        
        # we could index which data points are associated through data_table
        data_table = []

        for i in range(self.nmod):
            values,inv_id = np.unique(self.ind_tr[:,i],return_inverse=True)
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

    def moment_produc_U(self,modes,ind,mode='two'):
        # compute first and second moments of 
        # \kronecker_prod_{k \in given modes} u_k -Tucker
        # \Hadmard_prod_{k \in given modes} u_k -CP
        last_mode = modes[-1]

        E_z = self.post_U_m[last_mode][ind[:,last_mode]] # N*R_u*1
        # print(self.post_U_v[last_mode][ind[:,last_mode]].shape)
        E_z_2 = self.post_U_v[last_mode][ind[:,last_mode]] + torch.bmm(E_z,E_z.transpose(dim0=1,dim1=2)) # N*R_u*R_U

        for mode in reversed(modes[:-1]):
            E_u = self.post_U_m[mode][ind[:,mode]] # N*R_u*1
            E_u_2 = self.post_U_v[mode][ind[:,mode]] + torch.bmm(E_u,E_u.transpose(dim0=1,dim1=2)) # N*R_u*R_U

            E_z = self.batch_product(E_z,E_u)
            E_z_2 = self.batch_product(E_z_2,E_u_2)

        return E_z, E_z_2


    def expectation_update_z_del(self,del_mode):
        # compute E_z_del,E_z_del_2 by current post.U and deleting the info of mode_k
        
        other_modes = [i for i in range(self.nmod)]
        other_modes.remove(del_mode)
        self.E_z_del,self.E_z_del_2 = self.moment_produc_U(other_modes,self.ind_tr)

    def expectation_update_z(self): 
        # compute E_z,E_z_2 for given datapoints by current post.U (merge info of all modes )

        all_modes = [i for i in range(self.nmod)]        
        self.E_z,self.E_z_2 = self.moment_produc_U(all_modes,self.ind_tr) 

    def expectation_update_gamma(self):
        
        self.E_gamma= self.post_gamma_m # (R^K) * 1
        # self.E_gamma_2 = self.post_gamma_v + torch.mm(self.E_gamma,self.E_gamma.T) # (R^K)*(R^K)
        
    def expectation_update_tau(self):
        self.E_tau = self.post_a/self.post_b
    
    def msg_update_tau(self):

        self.msg_a = 1.5*self.ones_const

        term1 = 0.5 * torch.square(self.y_tr) # N*1

        term2 = self.y_tr * torch.matmul(self.E_z.transpose(dim0=1,dim1=2),self.E_gamma).squeeze(-1) # N*1
        
        temp = torch.matmul(self.E_z_2,self.E_gamma) # N*R^k* 1
        term3 = 0.5 * torch.matmul(temp.transpose(dim0=1,dim1=2),self.E_gamma).squeeze(-1) # N*1

        # alternative way to compute term3, where we have to compute and store E_gamma_2
        # term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1

        self.msg_b =  term1 - term2 + term3 # N*1

    def msg_update_U(self):
        for mode in range(self.nmod):
            
            self.expectation_update_z_del(mode)
            # self.expectation_update_z(self.ind_tr)

            E_gamma_del_mode = self.E_gamma_del_mode_func(mode)
            # print()

            for j in range(len(self.uid_table[mode])):
                uid = self.uid_table[mode][j] # id of embedding
                eid = self.data_table[mode][j] # id of associated entries
                
                # compute msg of associated entries, update with damping

                # some mid terms (to compute E_a_2 = W_k  * z\z\.T *  W_k.T)
                # print(E_gamma_del_mode.shape)
                term1 = torch.matmul(self.E_z_del_2[eid],E_gamma_del_mode.T) # num_eid * R_U^{K-1} * R_U
                term2 = torch.matmul(term1.transpose(dim0=1,dim1=2),E_gamma_del_mode.T).transpose(dim0=1,dim1=2)# num_eid * R_U * R_U

                msg_U_lam_new = self.E_tau * term2 # num_eid * R_U * R_U

                # the alternative way to compute msg_U_lam, but need E_gamma_2, non-effcient 
                # msg_U_lam_new = self.E_tau * self.E_z_del_2[eid] * self.E_gamma_2[eid] # num_eid * R_U * R_U
                
                # to compute E_a = W_k * z\ 
                term3 = torch.matmul(self.E_z_del[eid].transpose(dim0=1,dim1=2),E_gamma_del_mode.T).transpose(dim0=1,dim1=2)# num_eid * R_U * 1
                msg_U_eta_new =  self.E_tau * torch.bmm(term3, self.y_tr[eid].unsqueeze(-1)) # num_eid * R_U *1
                
                self.msg_U_lam[mode][eid] = self.DAMPPING_U * self.msg_U_lam[mode][eid] + (1- self.DAMPPING_U ) * msg_U_lam_new # num_eid * R_U * R_U
                self.msg_U_eta[mode][eid] = self.DAMPPING_U * self.msg_U_eta[mode][eid] + (1- self.DAMPPING_U ) * msg_U_eta_new # num_eid * R_U * 1
    
    def msg_update_gamma(self):
        
        msg_gamma_lam_new = self.E_tau*self.E_z_2 # N*(R^K)*(R^K)
        
        msg_gamma_eta_new = self.E_tau*torch.bmm(self.E_z,self.y_tr.unsqueeze(-1))# N*(R^K)*1

        self.msg_gamma_lam = self.DAMPPING_gamma * self.msg_gamma_lam + (1-self.DAMPPING_gamma)*msg_gamma_lam_new # N*(R^K)*(R^K)
        self.msg_gamma_eta = self.DAMPPING_gamma * self.msg_gamma_eta + (1-self.DAMPPING_gamma)*msg_gamma_eta_new # N*(R^K)*1

        # self.msg_gamma_v = torch.linalg.inv(self.msg_gamma_lam)
        # self.msg_gamma_m = torch.bmm(self.msg_gamma_v,self.msg_gamma_eta)        

    def post_update_U(self):
        
        # merge such msgs to get post.U

        for mode in range(self.nmod):
            for j in range(len(self.uid_table[mode])):

                uid = self.uid_table[mode][j] # id of embedding
                eid = self.data_table[mode][j] # id of associated entries

                self.post_U_v[mode][uid] = torch.linalg.inv(self.msg_U_lam[mode][eid].sum(dim=0) + (1.0/self.v)*self.eye_const) # R_U * R_U
                self.post_U_m[mode][uid] = torch.mm(self.post_U_v[mode][uid],self.msg_U_eta[mode][eid].sum(dim=0)) # R_U *1
 
    def post_update_tau(self):
        # update post. factor of tau based on current msg. factors
    
        self.post_a = self.a0 + self.msg_a.sum() - self.N
        self.post_b = self.b0 + self.msg_b.sum()  

    def post_update_gamma(self):
        
        self.post_gamma_v = torch.linalg.inv(self.msg_gamma_lam.sum(dim=0)) # (R^K) * (R^K)

        self.post_gamma_m = torch.mm(self.post_gamma_v,self.msg_gamma_eta.sum(dim=0))# (R^K) * 1

    def model_test(self,test_ind,test_y,test_time=None):
            
        MSE_loss = torch.nn.MSELoss()

        y_pred_train = torch.matmul(self.E_z.transpose(dim0=1,dim1=2),self.E_gamma).squeeze()
        y_true_train = self.y_tr.squeeze()
        loss_train =  torch.sqrt(MSE_loss(y_pred_train,y_true_train))

        all_modes = [i for i in range(self.nmod)]        
        E_z_test,_ = self.moment_produc_U(all_modes,test_ind) 

        y_pred_test = torch.matmul(E_z_test.transpose(dim0=1,dim1=2),self.E_gamma).squeeze()
        y_true_test = test_y.to(self.device).squeeze()
        loss_test =  torch.sqrt(MSE_loss(y_pred_test,y_true_test))

        return loss_train,loss_test

class static_CEP_CP(static_CEP_base):
    def __init__(self,hyper_para_dict):
        super(static_CEP_CP,self).__init__(hyper_para_dict)

        self.batch_product = Hadamard_product_batch
        self.E_gamma_del_mode_func = self.gamma_del_mode_func

        self.expectation_update_z()
        for mode in range(self.nmod):
            self.expectation_update_z_del(mode)        

    def gamma_del_mode_func(self,mode):
        return self.E_gamma.T

class static_CEP_Tucker_standard(static_CEP_base):

    def __init__(self,hyper_para_dict):
        super(static_CEP_Tucker_standard,self).__init__(hyper_para_dict)

        # self.msg_U_lam = [1e-4*torch.ones(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod) ] # (N*R_U*R_U)*nmod
        # self.msg_U_eta =  [torch.zeros(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*1)*nmod

        # self.msg_gamma_lam = 1e-4*torch.ones(self.N,self.gamma_size,1).double().to(self.device) # gamma_size *1
        # self.msg_gamma_eta = torch.zeros(self.N,self.gamma_size,1).double().to(self.device) # N*(R^K)*1


        self.batch_product = kronecker_product_einsum_batched
        self.E_gamma_del_mode_func = self.gamma_del_mode_func

        # # post. of gamma
        # self.post_gamma_m = torch.zeros(self.gamma_size,1).double().to(self.device) # (R^K)*1
        # self.post_gamma_v = torch.ones(self.gamma_size,1).double().to(self.device) # (R^K)*1

        # # # post.of U     
        # self.post_U_m = [item.unsqueeze(-1) for item in self.U] # nmod(list) * (ndim_i * R_U *1) 
        # self.post_U_v = [(self.v) *torch.ones(ndim,self.R_U,1).double().to(self.device) \
        #                 for ndim in self.ndims]# nmod(list) * (ndim_i * R_U *R_U ) 


        self.expectation_update_z()
        print(self.E_z.shape,self.E_z_2.shape)
        for mode in range(self.nmod):
            self.expectation_update_z_del(mode)


    def gamma_del_mode_func(self,mode):
        E_gamma_tensor = tl.tensor(self.E_gamma.reshape(self.nmod_list)) # (R^k *1)-> (R * R * R ...)
        E_gamma_mat_k = tl.unfold(E_gamma_tensor,mode).double()   #(R * R * R ...)-> -> (R * (R^{K-1}))
        return E_gamma_mat_k   

        
    # def moment_produc_U(self,modes,ind):
    #     # computhe first and second moments of z or z^{del}
    #     # \kronecker_prod_{k \in given modes} u_k -Tucker
    #     # \Hadmard_prod_{k \in given modes} u_k -CP
    #     last_mode = modes[-1]

    #     E_z = self.post_U_m[last_mode][ind[:,last_mode]] # N*gamma_size*1
    #     # E_z_2 = torch.diagonal(self.post_U_v[last_mode][ind[:,last_mode]],dim1=1, dim2=2).unsqueeze(-1) \
    #     #         + torch.square(E_z) # N*gamma_size*1
    #     E_z_2 = self.post_U_v[last_mode][ind[:,last_mode]]\
    #             + torch.square(E_z) # N*gamma_size*1

    #     for mode in reversed(modes[:-1]):
    #         E_u = self.post_U_m[mode][ind[:,mode]] # N*R_u*1
    #         # E_u_2 = torch.diagonal(self.post_U_v[mode][ind[:,mode]],dim1=1, dim2=2).unsqueeze(-1) + torch.square(E_u) # N*R_u*R_U
    #         E_u_2 = self.post_U_v[mode][ind[:,mode]] + torch.square(E_u) # N*R_u*R_U

    #         E_z = self.batch_product(E_z,E_u)
    #         E_z_2 = self.batch_product(E_z_2,E_u_2)

    #     return E_z, E_z_2
        
    # def msg_update_tau(self):
    
    #     self.msg_a = 1.5*self.ones_const

    #     term1 = 0.5 * torch.square(self.y_tr) # N*1

    #     term2 = self.y_tr * torch.matmul(self.E_z.transpose(dim0=1,dim1=2),self.E_gamma).squeeze(-1) # N*1
        
    #     temp = self.E_z_2 * self.E_gamma # N*R^k* 1
    #     term3 = 0.5 * torch.matmul(temp.transpose(dim0=1,dim1=2),self.E_gamma).squeeze(-1) # N*1

    #     # alternative way to compute term3, where we have to compute and store E_gamma_2
    #     # term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1

    #     self.msg_b =  term1 - term2 + term3 # N*1

    # def msg_update_U(self):
    #     for mode in range(self.nmod):
            
    #         self.expectation_update_z_del(mode)
    #         # self.expectation_update_z(self.ind_tr)

    #         E_gamma_del_mode = self.gamma_del_mode_func(mode)

    #         for j in range(len(self.uid_table[mode])):
    #             uid = self.uid_table[mode][j] # id of embedding
    #             eid = self.data_table[mode][j] # id of associated entries
                
    #             # compute msg of associated entries, update with damping

    #             # some mid terms to compute E_a_2 =  W_k  * (z\z\.T) *  W_k.T  / diag(E_a_2) 

    #             # first method: direct compute, with first order approx.
    #             # E_a_2 =  W_k  * (z\z\.T) *  W_k.T ~ W_k  * E(z\)E(z\.T) *  W_k.T
    #             # or diag(E_a_2) = diag( W_k  * (z\z\.T) *  W_k.T) ~ diag(W_k  * E(z\)E(z\.T) *  W_k.T)
    #             # as num_eid here is relative small, may work
                
    #             # E_z_del_2 = torch.bmm(self.E_z_del[eid], self.E_z_del[eid].transpose(dim0=1,dim1=2)) # num_eid * R_U^{K-1} * R_U^{K-1}
                
    #             E_z_del_2 = torch.diag_embed(self.E_z_del_2[eid].squeeze(-1))
    #             # print(E_z_del_2.size())
                
    #             term1 = torch.matmul(E_z_del_2,E_gamma_del_mode.T) # num_eid * R_U^{K-1} * R_U
                
    #             term2 = torch.matmul(term1.transpose(dim0=1,dim1=2),E_gamma_del_mode.T).transpose(dim0=1,dim1=2)# num_eid * R_U * R_U
    #             # msg_U_lam_new = self.E_tau * term2
    #             msg_U_lam_new = self.E_tau * torch.diagonal(term2,dim1=1, dim2=2).unsqueeze(-1) # num_eid * R_U * 1

    #             # second method: very raw approx.
    #             # diag(E_a_2) = diag( W_k  * (z\z\.T) *  W_k.T)
    #             # ~ diag( W_k  * diag(z\z\.T) *  W_k.T) ~ matmul ( (W_k)^2 , diag(z\z\.T))
                
    #             # term1 = torch.matmul(E_gamma_del_mode,self.E_z_del_2[eid]) # num_eid * R_U * 1
    #             # msg_U_lam_new = self.E_tau * term1
                
    #             # to compute E_a = W_k * z\ 
    #             term3 = torch.matmul(self.E_z_del[eid].transpose(dim0=1,dim1=2),E_gamma_del_mode.T).transpose(dim0=1,dim1=2)# num_eid * R_U * 1
    #             msg_U_eta_new =  self.E_tau * torch.bmm(term3, self.y_tr[eid].unsqueeze(-1)) # num_eid * R_U *1
                
    #             self.msg_U_lam[mode][eid] = self.DAMPPING_U * self.msg_U_lam[mode][eid] + (1- self.DAMPPING_U ) * msg_U_lam_new # num_eid * R_U * R_U
    #             self.msg_U_eta[mode][eid] = self.DAMPPING_U * self.msg_U_eta[mode][eid] + (1- self.DAMPPING_U ) * msg_U_eta_new # num_eid * R_U * 1

    # def msg_update_gamma(self):
        
    #     msg_gamma_lam_new = self.E_tau*self.E_z_2 # N*(R^K)*1
    #     msg_gamma_eta_new = self.E_tau*torch.bmm(self.E_z,self.y_tr.unsqueeze(-1))# N*(R^K)*1

    #     self.msg_gamma_lam = self.DAMPPING_gamma * self.msg_gamma_lam + (1-self.DAMPPING_gamma)*msg_gamma_lam_new # N*(R^K)*(R^K)
    #     self.msg_gamma_eta = self.DAMPPING_gamma * self.msg_gamma_eta + (1-self.DAMPPING_gamma)*msg_gamma_eta_new # N*(R^K)*1

    # def post_update_gamma(self):
        
    #     self.post_gamma_v = 1.0/(self.msg_gamma_lam.sum(dim=0)) # (R^K) * 1
    #     self.post_gamma_m = self.post_gamma_v * self.msg_gamma_eta.sum(dim=0)# (R^K) * 1

    # def post_update_U(self):
        
    #     # merge such msgs to get post.U

    #     for mode in range(self.nmod):
    #         for j in range(len(self.uid_table[mode])):

    #             uid = self.uid_table[mode][j] # id of embedding
    #             eid = self.data_table[mode][j] # id of associated entries

    #             self.post_U_v[mode][uid] = 1.0/(self.msg_U_lam[mode][eid].sum(dim=0)) # R_U * R_U
    #             self.post_U_m[mode][uid] = self.post_U_v[mode][uid]*self.msg_U_eta[mode][eid].sum(dim=0) # R_U *1
 