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

class  LDS_dynammic_CP():
    def __init__(self,hyper_para_dict,data):

        super(dynamic_CP,self).__init__(hyper_para_dict,data)
        
        # data 
        self.data

        self.ndims = 

        # if kernel is matern-1.5, factor = 1, kernel is matern-2.5, factor =2    
        self.FACTOR = 1

        hyper_para_dict_list

        # prior of noise
        self.v = hyper_para_dict['v'] # prior varience of embedding (scaler)
        self.a0 = hyper_para_dict['a0']
        self.b0 = hyper_para_dict['b0']

        # dynamic for each mode
        self.LDS_list = [LDS_GP(hyper_para_dict_list[i]) for i in range(self.nmods)]

        # posterior 
        self.post_U_m = [torch.zeros(dim,self.R_U,1,self.N_time).double().to(self.device) for dim in range(self.ndims)] #  (dim, R_U, 1, T) * nmod
        self.post_U_v = [torch.eye(self.R_U).reshape((1,self.R_U,self.R_U,1)).repeat(dim,1,1,self.N_time).double().to(self.device) for dim in range(self.ndims)] # (dim, R_U, R_U, T) * nmod

        # msg (long vec, place-holder, update by current posterior before use)
        self.msg_U_M = None
        self.msg_U_V = None

        # some constant terms 
        self.ones_const = torch.ones(self.N,1).to(self.device)

        # build time-data table: Given a time-stamp id, return the indexed of entries
        self.time_data_table_tr = utils.build_time_data_table(self.train_time_ind)
        self.time_data_table_te = utils.build_time_data_table(self.test_time_ind)

    def msg_update_U(self,mode,T):
        # init the msg_U_M, msg_U_V
        size_long_vec = self.R_U*self.ndims[mode]*self.FACTOR
        self.msg_U_M = torch.zeros(size_long_vec,1).to(self.device)
        self.msg_U_V = torch.diag(1e6*torch.ones(size_long_vec)).to(self.device)

        # retrive the observed entries at T
        eind_T = self.time_data_table_tr[T]  # list of observed entries id at this time-stamp
        N_T = len(eind_T)
        ind_T = self.ind_tr[eind_T]
        y_T = self.y_tr[eind_T].squeeze()

        condi_modes = [i for i in range(self.nmod)].remove(mode)

        uid_table, data_table = utils.build_id_key_table(
            nmod = 1, ind = ind_T[:,mode]
        )  # get the id of associated nodes at current mode

        E_z, E_z_2 = utils.moment_Hadmard(
                                            modes=condi_modes,\
                                            ind = ind_T,\
                                            U_m = [ele[:,:,:,T] for ele in self.post_U_m],\
                                            U_v = [ele[:,:,:,T] for ele in self.post_U_v],\
                                            sum_2_scaler=True,\
                                            device=self.device
                                            )
        
        # use the nature-paras first, convinient to merge msg later 
        S_inv = self.E_tau * E_z_2 # (N,R,R)
        S_inv_Beta = y_T * E_z # (N,R,1)

        # filling the msg_U_M, msg_U_V
        for i in range(len(uid_table)):
            uid = uid_table[i] # id of embedding
            eid = data_table[i] # id of associated entries

            idx_start = uid*self.R_U
            idx_end = (uid+1)*self.R_U

            U_V = torch.linalg.inv(S_inv.sum(dim=0) + (1.0/self.v)*torch.eye(self.R_U).to(self.device)) # (R,R)
            U_M = torch.mm(U_V,S_inv_Beta.sum(dim=0)) # (R,1)

            self.msg_U_V[idx_start:idx_end,idx_start:idx_end] = U_V
            self.msg_U_M[idx_start:idx_end] = U_M
    
    def post_update_U(self,mode):
        # update post. of U based on latest results from RTS-smoother
        
        LDS = self.LDS_list[mode]
        H = LDS.H
        dim = self.ndims[mode]

        for t in range(self.N_time):
            vec_U_M = torch.mm(H,LDS.m_smooth_list[t]) # (dim*R_U,1)
            vec_U_V = torch.mm(H, torch.mm(LDS.m_smooth_list[t],H.T) # (dim*R_U,dim*R_U)

            self.post_U_m[mode][:,:,0,t] = vec_U_M.reshape(dim,self.R_U)
            # self.post_U_m[mode][:,:,:,t] = vec_U_M.reshape(dim,self.R_U) # hard to extract block mats, use for-loop

            for j in range(dim):

                idx_start = j*self.R_U
                idx_start = (j+1)*self.R_U
                self.post_U_m[mode][j,:,:,t] = vec_U_V(idx_start:idx_end,idx_start:idx_end)


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