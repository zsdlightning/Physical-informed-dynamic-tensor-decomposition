'''
    File name: model_static_CP_Tucker.py
    Author: Shikai/Xuangu Fang
    Date created: 12/29/2021
    Locate: Shanghai,CN & SLC,US
    Python Version: 3
'''
'''
SVI based static CP/tucker decomposition, for scalability, point-estimate for gamma(weight part)
As the baselines of BCTT/LDS-Tucker paper, add time-mode-transition prior
'''


import numpy as np
from numpy.core.fromnumeric import transpose
import torch 
# import matplotlib.pyplot as plt

from utils import kronecker_product_einsum_batched,Hadamard_product_batch
# import tensorly as tl
# tl.set_backend('pytorch')

class static_SVI_base():
    '''
    base class for standard SVI update of CP/full-tucker model
    gamma: here refers lambda(weight vector) in CP or vec(W)(core-tensor) in Tucker
    z or z\: here refers \hadamard_product U_k in CP or \kronecker_product U_k
    U,tau: embedding & inverse-noise
    '''
    
    def __init__(self,hyper_para_dict):
        
        self.device = hyper_para_dict['device'] # add the cuda version later 


        # training data
        self.ind_tr = hyper_para_dict['tr_ind']
        self.y_tr = torch.tensor(hyper_para_dict['tr_y']).to(self.device) # N*1
        self.N = len(self.y_tr) # number of data-llk
        
        # some hyper-paras
        self.U = [torch.tensor(item,requires_grad=True,device=self.device) for item in hyper_para_dict['U']] # list of mode embedding, fixed and known in this setting

        self.nmod = len(self.U)
        self.R_U = hyper_para_dict['R_U'] # rank of latent factor of embedding

        self.ndims = hyper_para_dict['ndims'] 
        self.all_modes = [i for i in range(self.nmod)]


        self.nmod_list = [self.R_U for k in range(self.nmod)]
        self.gamma_size = hyper_para_dict['gamma_size'] # R_U for CP, (R_U)^K for tucker

        self.gamma = torch.ones((self.gamma_size,1),requires_grad=True,device=self.device, dtype=torch.double)

        # prior of noise
        self.v = hyper_para_dict['v'] # prior varience of embedding (scaler)
        self.v_time = hyper_para_dict['v_time'] # prior varience of time-mode embedding (scaler)

        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.jitter = torch.tensor(1e-4, device=self.device)

        self.para_list = self.U + [self.gamma, self.log_tau]


        

        # batch-product function: 
        # assgain as \kronecker_product in Tucker, \Hadamard_product
        self.batch_product = None

        

    def moment_produc_U(self,ind):
        # computhe first and second moments of 
        # \kronecker_prod_{k \in given modes} u_k -Tucker
        # \Hadmard_prod_{k \in given modes} u_k -CP
        last_mode = self.all_modes[-1]
        # print(ind.shape)
        E_z = self.U[last_mode][ind[:,last_mode]] # N*R_u

        for mode in reversed(self.all_modes[:-1]):
            E_u = self.U[mode][ind[:,mode]] # N*R_u
            E_z = self.batch_product(E_z,E_u)

        return E_z

    def nELBO_batch(self,batch_ind):

        B_size = batch_ind.shape[0]
        ind_x = self.ind_tr[batch_ind]

        y_pred = self.pred(ind_x).squeeze() 
        y_true = self.y_tr[batch_ind].squeeze()

        tau = torch.exp(self.log_tau)
        ELBO = 0.5*self.N*self.log_tau \
            - 0.5*tau*self.N/B_size*torch.sum(torch.square(y_true - y_pred))

        return -torch.squeeze(ELBO)

    def pred(self,x_ind):
        E_z = self.moment_produc_U(x_ind).squeeze(-1) # N* gamma_size
        pred_y = torch.mm(E_z,self.gamma) # N*1
        return pred_y

    def model_test(self,test_ind,test_y):
        with torch.no_grad():    

            MSE_loss = torch.nn.MSELoss()
            MAE_loss = torch.nn.L1Loss()


            y_pred_train = self.pred(self.ind_tr).squeeze() # N*1
            y_true_train = self.y_tr.squeeze()
            loss_train =  torch.sqrt(MSE_loss(y_pred_train,y_true_train))

            y_pred_test = self.pred(test_ind).squeeze()
            y_true_test = test_y.to(self.device).squeeze()
            loss_test_rmse =  torch.sqrt(MSE_loss(y_pred_test,y_true_test))
            loss_test_MAE = MAE_loss(y_pred_test,y_true_test)

            # return loss_train,loss_test
            return loss_train,loss_test_rmse,loss_test_MAE


class transition_SVI_base(static_SVI_base):
    def __init__(self,hyper_para_dict):
        super(transition_SVI_base,self).__init__(hyper_para_dict)

        self.W = torch.zeros([self.R_U,self.R_U], device=self.device, requires_grad=True)
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.zeros(self.R_U, device=self.device, requires_grad=True)
        self.log_v = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.para_list = self.U + [self.gamma, self.log_tau, self.W,self.b,self.log_v]

    def trans_prior(self,):
        T = self.U[-1].squeeze().float()
        # print(T.shape)
        trans_mu = torch.tanh(torch.matmul(T, self.W) + self.b)
        I = torch.eye(T.shape[1]).to(self.device)
        trans_std = torch.exp(self.log_v)*I
        
        T = T[1:, :]
        trans_mu = trans_mu[:-1, :]
        
        prior_dist = torch.distributions.MultivariateNormal(loc=trans_mu, covariance_matrix=trans_std)
        
        log_prior = prior_dist.log_prob(T)

        return log_prior.sum()

    def nELBO_batch(self, batch_ind):
        B_size = batch_ind.shape[0]
        ind_x = self.ind_tr[batch_ind]

        y_pred = self.pred(ind_x).squeeze() 
        y_true = self.y_tr[batch_ind].squeeze()
        tau = torch.exp(self.log_tau)
        
        trans_log_prob = self.trans_prior()
        
        ELBO = 0.5*self.N*tau -0.5*torch.exp(self.log_tau)*self.N/B_size*torch.sum(torch.square(y_pred - y_true)) +\
        trans_log_prob
 
        return -torch.squeeze(ELBO)


class static_SVI_CP(static_SVI_base):
    def __init__(self,hyper_para_dict):
        super(static_SVI_CP,self).__init__(hyper_para_dict)

        self.batch_product = Hadamard_product_batch   


class static_SVI_Tucker(static_SVI_base):

    def __init__(self,hyper_para_dict):
        super(static_SVI_Tucker,self).__init__(hyper_para_dict)

        self.batch_product = kronecker_product_einsum_batched

class transition_SVI_CP(transition_SVI_base):
    def __init__(self,hyper_para_dict):
        super(transition_SVI_CP,self).__init__(hyper_para_dict)

        self.batch_product = Hadamard_product_batch  
  


class transition_SVI_Tucker(transition_SVI_base):

    def __init__(self,hyper_para_dict):
        super(transition_SVI_Tucker,self).__init__(hyper_para_dict)

        self.batch_product = kronecker_product_einsum_batched