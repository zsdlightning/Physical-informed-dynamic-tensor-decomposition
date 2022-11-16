"""
Implementation of Streaming Factor Trajectory for Dynamic Tensor, current is CP version, to be extend to Tucker 

The key differences of the idea and current one is: 
1. Build independent Trajectory Class (LDS-GP) for each embedding
2. Streaming update (one (batch) llk -> multi-msg to multi LDS -> filter_update simultaneously-> finally smooth back) 

draft link: https://www.overleaf.com/project/6363a960485a46499baef800
Authod: Shikai Fang
SLC, Utah, 2022.11
"""

import numpy as np
from numpy.lib import utils
import torch
import matplotlib.pyplot as plt
from model_LDS import LDS_GP_streaming
import os
import tqdm
import utils_streaming

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
JITTER = 1e-4

torch.manual_seed(2)


class LDS_dynammic_streaming:
    def __init__(self, hyper_dict, data_dict):

        """-----------------hyper-paras---------------------"""
        self.device = hyper_dict["device"]
        self.R_U = hyper_dict["R_U"]  # rank of latent factor of embedding

        # prior of noise
        self.v = hyper_dict["v"]  # prior varience of embedding (scaler)
        self.a0 = hyper_dict["a0"]
        self.b0 = hyper_dict["b0"]
        self.FACTOR = hyper_dict["FACTOR"]

        """----------------data-dependent paras------------------"""
        # if kernel is matern-1.5, factor = 1, kernel is matern-2.5, factor =2

        self.ndims = data_dict["ndims"]
        self.nmods = len(self.ndims)

        self.ind_tr = data_dict["ind_tr"]
        self.y_tr = torch.tensor(data_dict["y_tr"]).to(self.device)  # N*1

        self.ind_te = data_dict["ind_te"]
        self.y_te = torch.tensor(data_dict["y_te"]).to(self.device)  # N*1

        self.train_time_ind = data_dict["T_disct_tr"]  # N_train*1
        self.test_time_ind = data_dict["T_disct_te"]  # N_test*1

        self.time_uni = data_dict["time_uni"]  # N_time*1
        self.N_time = len(self.time_uni)

        # build dynamics (LDS-GP class) for each object in each mode (store using nested list)
        self.traj_class = []
        for mode in range(self.nmods):
            traj_class_mode = [
                LDS_GP_streaming(hyper_dict) for i in range(self.ndims[mode])
            ]
            self.traj_class.append(traj_class_mode)

        # posterior: store the most recently posterior from LDS for fast test?
        self.post_U_m = [
            torch.randn(dim, self.R_U, 1, self.N_time).double().to(self.device)
            for dim in self.ndims
        ]  #  (dim, R_U, 1, T) * nmod
        self.post_U_v = [
            torch.eye(self.R_U)
            .reshape((1, self.R_U, self.R_U, 1))
            .repeat(dim, 1, 1, self.N_time)
            .double()
            .to(self.device)
            for dim in self.ndims
        ]  # (dim, R_U, R_U, T) * nmod

        self.E_tau = 1.0

        # msg (long vec, place-holder, update by current posterior before use)
        self.msg_U_M = None
        self.msg_U_V = None

        # build time-data table: Given a time-stamp id, return the indexed of entries
        self.time_data_table_tr = utils_streaming.build_time_data_table(
            self.train_time_ind
        )
        self.time_data_table_te = utils_streaming.build_time_data_table(
            self.test_time_ind
        )

        # place holder, will be updated at (track_envloved_object) at each time step
        self.ind_T = None
        self.y_T = None
        self.uid_table = None
        self.data_table = None

    def track_envloved_objects(self, T):

        """retrive the index/values/object-id of observed entries at T"""

        eind_T = self.time_data_table_tr[
            T
        ]  # list of observed entries id at this time-stamp

        self.ind_T = self.ind_tr[eind_T]
        self.y_T = self.y_tr[eind_T].reshape(-1, 1, 1)

        self.uid_table, self.data_table = utils_streaming.build_id_key_table(
            nmod=self.nmods, ind=self.ind_T
        )  # nested-list of observed objects (and their associated entrie) at this time-stamp

    def filter_predict(self, T):

        """trajectories of involved objects take KF prediction step + update the posterior"""

        current_time_stamp = self.time_uni[T]

        for mode in range(self.nmods):
            for uid in self.uid_table[mode]:
                self.traj_class[mode][uid].filter_predict(current_time_stamp)

                # update the posterior based on the prediction state
                """double check, need mul observed-mat to change shape"""
                self.post_U_m[mode][uid, :, :, T] = self.traj_class[mode][
                    uid
                ].m_pred_list[-1]

                self.post_U_v[mode][uid, :, :, T] = self.traj_class[mode][
                    uid
                ].P_pred_list[-1]

    def msg_approx(self, T):
        # init the msg_U_M, msg_U_V
        size_long_vec = self.R_U * self.ndims[mode]  # * self.FACTOR
        self.msg_U_M = torch.zeros(size_long_vec, 1).to(self.device)
        self.msg_U_V = torch.diag(1e6 * torch.ones(size_long_vec)).to(self.device)

        # retrive the observed entries at T
        eind_T = self.time_data_table_tr[
            T
        ]  # list of observed entries id at this time-stamp
        N_T = len(eind_T)
        ind_T = self.ind_tr[eind_T]
        y_T = self.y_tr[eind_T].reshape(-1, 1, 1)

        condi_modes = [i for i in range(self.nmods)]
        condi_modes.remove(mode)

        uid_table, data_table = utils_streaming.build_id_key_table(
            nmod=1, ind=ind_T[:, mode]
        )  # get the id of associated nodes at current mode

        E_z, E_z_2 = utils_streaming.moment_Hadmard(
            modes=condi_modes,
            ind=ind_T,
            U_m=[ele[:, :, :, T] for ele in self.post_U_m],
            U_v=[ele[:, :, :, T] for ele in self.post_U_v],
            order="second",
            sum_2_scaler=False,
            device=self.device,
        )

        # use the nature-paras first, convinient to merge msg later
        S_inv = self.E_tau * E_z_2  # (N,R,R)
        S_inv_Beta = y_T * E_z  # (N,R,1)

        # filling the msg_U_M, msg_U_V
        for i in range(len(uid_table)):
            uid = uid_table[i]  # id of embedding
            eid = data_table[i]  # id of associated entries

            idx_start = uid * self.R_U
            idx_end = (uid + 1) * self.R_U

            U_V = torch.linalg.inv(
                S_inv[eid].sum(dim=0)
                + (1.0 / self.v) * torch.eye(self.R_U).to(self.device)
            )  # (R,R)
            U_M = torch.mm(U_V, S_inv_Beta[eid].sum(dim=0))  # (R,1)

            self.msg_U_V[idx_start:idx_end, idx_start:idx_end] = U_V
            self.msg_U_M[idx_start:idx_end] = U_M

    def post_update_U(self, mode):
        # update post. of U based on latest results from RTS-smoother

        LDS = self.LDS_list[mode]
        H = LDS.H
        dim = self.ndims[mode]

        for t in range(self.N_time):
            vec_U_M = torch.mm(H, LDS.m_smooth_list[t])  # (dim*R_U,1)
            vec_U_V = torch.mm(
                H, torch.mm(LDS.P_smooth_list[t], H.T)
            )  # (dim*R_U,dim*R_U)

            self.post_U_m[mode][:, :, 0, t] = vec_U_M.reshape(dim, self.R_U)
            # self.post_U_v[mode][:,:,:,t] = vec_U_M.reshape(dim,self.R_U) # hard to extract block mats, use for-loop

            for j in range(dim):

                idx_start = j * self.R_U
                idx_end = (j + 1) * self.R_U
                self.post_U_v[mode][j, :, :, t] = vec_U_V[
                    idx_start:idx_end, idx_start:idx_end
                ]

    def model_test(self, test_ind, test_y, test_time):

        MSE_loss = torch.nn.MSELoss()
        all_modes = [i for i in range(self.nmods)]

        tid = test_time

        pred = utils_streaming.moment_Hadmard_T(
            modes=all_modes,
            ind=test_ind,
            ind_T=tid,
            U_m_T=self.post_U_m,
            U_v_T=self.post_U_v,
            order="first",
            sum_2_scaler=True,
            device=self.device,
        )

        loss_test_base = torch.sqrt(
            MSE_loss(pred.squeeze(), test_y.squeeze().to(self.device))
        )

        return loss_test_base
