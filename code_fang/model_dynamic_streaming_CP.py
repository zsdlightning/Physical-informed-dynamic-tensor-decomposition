"""
Implementation of Streaming Factor Trajectory for Dynamic Tensor, current is CP version, to be extended to Tucker 

The key differences of the idea and current one is: 
1. Build independent Trajectory Class (LDS-GP) for each embedding
2. Streaming update (one (batch) llk -> multi-msg to multi LDS -> filter_update simultaneously-> finally smooth back) 

draft link: https://www.overleaf.com/project/6363a960485a46499baef800
Author: Shikai Fang
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
import bisect

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

        # place holders
        self.ind_T = None
        self.y_T = None
        self.uid_table = None
        self.data_table = None

        self.msg_U_m = None
        self.msg_U_V = None

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
                H = self.traj_class[mode][uid].H
                m = self.traj_class[mode][uid].m_pred_list[-1]
                P = self.traj_class[mode][uid].P_pred_list[-1]
                self.post_U_m[mode][uid, :, :, T] = torch.mm(H, m)
                self.post_U_v[mode][uid, :, :, T] = torch.mm(torch.mm(H, P), H.T)

    def msg_approx(self, T):
        """approx the msg from the group of data-llk at T"""

        # reste msg_U_m, msg_U_V

        self.msg_U_m = []
        self.msg_U_V = []

        for mode in range(self.nmods):
            msg_U_m_mode = []
            msg_U_V_mode = []

            condi_modes = [i for i in range(self.nmods)]
            condi_modes.remove(mode)
            E_z, E_z_2 = utils_streaming.moment_Hadmard(
                modes=condi_modes,
                ind=self.ind_T,
                U_m=[ele[:, :, :, T] for ele in self.post_U_m],
                U_v=[ele[:, :, :, T] for ele in self.post_U_v],
                order="second",
                sum_2_scaler=False,
                device=self.device,
            )

            # use the nature-paras first, convinient to merge msg later
            S_inv = self.E_tau * E_z_2  # (N,R,R)
            S_inv_Beta = self.y_T * E_z  # (N,R,1)

            # filling the msg_U_M, msg_U_V
            for i in range(len(self.uid_table[mode])):
                uid = self.uid_table[mode][i]  # id of embedding
                eid = self.data_table[mode][i]  # id of associated entries

                U_V = torch.linalg.inv(
                    S_inv[eid].sum(dim=0)
                    + (1.0 / self.v) * torch.eye(self.R_U).to(self.device)
                )  # (R,R)
                U_M = torch.mm(U_V, S_inv_Beta[eid].sum(dim=0))  # (R,1)

                msg_U_m_mode.append(U_M)
                msg_U_V_mode.appned(U_V)

            self.msg_U_m.append(msg_U_m_mode)
            self.msg_U_V.append(msg_U_V_mode)

    def filter_update(self, T):
        """trajectories of involved objects take KF update step"""
        for mode in range(self.nmods):
            for msg_id, uid in enumerate(self.uid_table[mode]):

                # we treat the approx msg as the observation values for KF
                y = self.msg_U_m[mode][msg_id]
                R = self.msg_U_V[mode][msg_id]

                # KF update step
                self.traj_class[mode][uid].filter_update(y=y, R=R)

                # update the posterior? -- no need

    def smooth(self):
        """smooth back for all objects"""
        for mode in range(self.nmods):
            for uid in range(self.ndims[mode]):
                self.traj_class[mode][uid].smooth()

    def get_post_U(self):
        """get the final post of U using the smoothed result"""
        for T, time_stamp in enumerate(self.time_uni):
            for mode in range(self.nmods):
                for uid in range(self.ndims[mode]):
                    traj = self.traj_class[mode][uid]

                    if time_stamp in traj.time_stamp_list:
                        # the time_stamp appread before

                        T_id = traj.time_2_ind_table[time_stamp]
                        # update the posterior based on the smoothed state

                        H = traj.H
                        m = traj.m_smooth_list[T_id]
                        P = traj.P_smooth_list[T_id]

                        self.post_U_m[mode][uid, :, :, T] = torch.mm(H, m)
                        self.post_U_v[mode][uid, :, :, T] = torch.mm(
                            torch.mm(H, P), H.T
                        )

                    else:
                        # the time_stamp never appread before
                        print("the time_stamp never appread before")

                        # locate the place of un-seen time_stamp
                        loc = bisect.bisect(traj.time_stamp_list, time_stamp)

                        if loc < len(traj.time_stamp_list):
                            # interpolation, merge (follow formulas 10-13 in draft)

                            prev_time_stamp = traj.time_stamp_list[loc - 1]
                            next_time_stamp = traj.time_stamp_list[loc]

                            prev_m = traj.m_smooth_list[loc - 1]
                            prev_P = traj.P_smooth_list[loc - 1]

                            next_m = traj.m_smooth_list[loc]
                            next_P = traj.P_smooth_list[loc]

                            prev_time_int = time_stamp - prev_time_stamp
                            next_time_int = next_time_stamp - time_stamp

                            prev_A = torch.matrix_exp(traj.F * prev_time_int).double()
                            prev_Q = traj.P_inf - torch.mm(
                                torch.mm(prev_A, traj.P_inf), prev_A.T
                            )

                            Q1_inv = torch.inverse(
                                torch.mm(torch.mm(prev_A, prev_P), prev_A.T) + prev_Q
                            )

                            next_A = torch.matrix_exp(traj.F * next_time_int).double()
                            next_Q = traj.P_inf - torch.mm(
                                torch.mm(next_A, traj.P_inf), next_A.T
                            )

                            Q2_inv = torch.inverse(
                                torch.mm(torch.mm(next_A, next_P), next_A.T) + next_Q
                            )

                            merge_P = torch.inverse(
                                Q1_inv + torch.mm(next_A.T, torch.mm(Q2_inv, next_A))
                            )

                            temp_term = torch.mm(
                                Q1_inv, torch.mm(prev_A, prev_m)
                            ) + torch.mm(Q2_inv, torch.mm(next_A, next_m))
                            merge_m = torch.mm(merge_P, temp_term)

                            H = traj.H
                            self.post_U_m[mode][uid, :, :, T] = torch.mm(H, merge_m)
                            self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                torch.mm(H, merge_P), H.T
                            )

                        else:
                            # extrapolation, gauss jump
                            prev_time_stamp = traj.time_stamp_list[loc - 1]
                            prev_m = traj.m_smooth_list[loc - 1]
                            prev_P = traj.P_smooth_list[loc - 1]
                            prev_time_int = time_stamp - prev_time_stamp

                            prev_A = torch.matrix_exp(traj.F * prev_time_int).double()
                            prev_Q = traj.P_inf - torch.mm(
                                torch.mm(prev_A, traj.P_inf), prev_A.T
                            )

                            jump_m = torch.mm(prev_A, prev_m)
                            jump_P = torch.inverse(
                                torch.mm(torch.mm(prev_A, prev_P), prev_A.T) + prev_Q
                            )

                            H = traj.H
                            self.post_U_m[mode][uid, :, :, T] = torch.mm(H, jump_m)
                            self.post_U_v[mode][uid, :, :, T] = torch.mm(
                                torch.mm(H, jump_P), H.T
                            )

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
