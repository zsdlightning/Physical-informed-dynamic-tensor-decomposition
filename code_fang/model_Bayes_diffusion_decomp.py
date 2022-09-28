import numpy as np
import scipy

# import pandas
import torch
import utils
from utils import generate_state_space_Matern_23
from scipy import linalg
from utils import build_id_key_table, moment_Hadmard

'''
the decompose CP-form of dynamic tensor
U(t) = CP ( U0 \circ Gamma(t) ) -- Gamma(t) size ?
-- same with U0? (num_node * R, diag_var )  -> try this first
-- with size num_node * 1 
U0: base embedding, update with standard CEP
Gamma(t): dynamic weighting, update by mixup state-space-model inspired by graph-diffusion:
-- ADF/CEP -base message passing update first.  -> try this first
-- KF/RTS
'''


class Bayes_diffu_tensor_decomp:
    def __init__(self, data_dict, hyper_dict):

        # hyper-paras
        self.epoch = hyper_dict["epoch"]  # passing epoch
        self.R_U = hyper_dict["R_U"]  # rank of latent factor of embedding
        self.device = hyper_dict["device"]
        self.DAMPING = hyper_dict["DAMPING"]

        self.a0 = hyper_dict["a0"]
        self.b0 = hyper_dict["b0"]

        self.m0 = torch.tensor(1.0)
        self.v0 = torch.tensor(1e1) # sentitive variable

        # data-dependent paras
        self.data_dict = data_dict

        self.ind_tr = data_dict["tr_ind"]
        self.y_tr = torch.tensor(data_dict["tr_y"]).to(self.device)  # N*1
        self.N_tr = len(self.y_tr)

        self.ind_te = data_dict["te_ind"]
        self.y_te = torch.tensor(data_dict["te_y"]).to(self.device)  # N*1

        self.N = len(data_dict["tr_y"])

        self.ndims = data_dict["ndims"]
        self.nmod = len(self.ndims)
        self.num_nodes = sum(self.ndims)

        self.train_time_ind = data_dict["tr_T_disct"]  # N*1
        self.test_time_ind = data_dict["te_T_disct"]  # N*1

        self.time_uni = data_dict["time_uni"]  # N_time*1
        self.N_time = len(self.time_uni)

        self.time_id_table = data_dict["time_id_table"]
        self.F = data_dict["F"].to(self.device)  # transition matrix
        self.P_inf = data_dict["P_inf"].to(self.device)  # stationary covar

        # init the message factor of llk term (U_llk, tau)
        # and transition term (U_f: U_forard, U_b: U_backward)

        # actually, it's the massage from llk-factor -> variabel Gamma

        self.msg_llk_2_Gamma_m = self.m0 * torch.rand(
            self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)
        self.msg_llk_2_Gamma_v = self.v0 * torch.ones(
            self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)

        self.msg_tau_a = torch.ones(self.N_time, 1).to(self.device)
        self.msg_tau_b = torch.ones(self.N_time, 1).to(self.device)

        # actually, it's the massage from transition-factor -> variabel Gamma
        # recall, there two kinds of msg: forward and backward
        # for here, we arrange the U-msg by concat-all-as-tensor for efficient computing in transition
        # recall, with Matern 23 kernel, msg_U_transition = [ U, U'], so the firsr-dim is 2*num_nodes

        self.msg_Trans_f_Gamma_m = self.m0 * torch.rand(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)
        self.msg_Trans_f_Gamma_v = self.v0 * torch.ones(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)

        self.msg_Trans_b_Gamma_m = self.m0 * torch.rand(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)
        self.msg_Trans_b_Gamma_v = self.v0 * torch.ones(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)

        # set the start and end factor

        for r in range(self.R_U):
            self.msg_Trans_b_Gamma_m[:, r, self.N_time - 1] = 0
            self.msg_Trans_b_Gamma_v[:, r, self.N_time - 1] = 1e8

            self.msg_Trans_f_Gamma_m[:, r, 0] = 0
            self.msg_Trans_f_Gamma_v[:, r, 0] = 1e8  # torch.diag(self.P_inf)

        # init the calibrating factors / q_del in draft, init/update with current msg

        # actually, it's the massage from variabel Gamma -> llk-factor
        self.msg_Gamma_2_llk_m = (
            torch.rand(self.num_nodes, self.R_U, self.N_time).double().to(self.device)
        )
        self.msg_Gamma_2_llk_v = self.v0 * torch.ones(
            self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)


        self.msg_tau_a_del_T = torch.ones(self.N_time, 1).to(self.device)
        self.msg_tau_b_del_T = torch.ones(self.N_time, 1).to(self.device)


        # actually, it's the massage from variabel Gamma -> trans-factor
        # recall, there two kinds of msg: forward and backward
        self.msg_Gamma_f_Trans_m = self.m0 * torch.rand(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)
        self.msg_Gamma_f_Trans_v = self.v0 * torch.ones(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)

        self.msg_Gamma_b_Trans_m = self.m0 * torch.rand(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)
        self.msg_Gamma_b_Trans_m = self.v0 * torch.ones(
            2 * self.num_nodes, self.R_U, self.N_time
        ).double().to(self.device)


        # init the message of U over each training data (using natural paras) 
        self.msg_U_lam = [1e-4*torch.eye(self.R_U).reshape((1,self.R_U,self.R_U)).repeat(self.N,1,1).double().to(self.device) for i in range(self.nmod) ] # (N*R_U*R_U)*nmod
        self.msg_U_eta =  [torch.zeros(self.N,self.R_U,1).double().to(self.device) for i in range(self.nmod)] # (N*R_U*1)*nmod

        # init the post. U and post. Gamma

        self.post_U_m = [
            torch.rand(ndim, self.R_U, 1).double().to(self.device)
            for ndim in self.ndims
        ]

        self.post_U_v = [
            torch.ones(ndim, self.R_U, 1).double().to(self.device)
            for ndim in self.ndims
        ]

        self.post_Gamma_m = [
            torch.rand(ndim, self.R_U, self.N_time).double().to(self.device)
            for ndim in self.ndims
        ]

        self.post_Gamma_v = [
            torch.ones(ndim, self.R_U, self.N_time).double().to(self.device)
            for ndim in self.ndims
        ]

        # time-data table
        # Given a time id, return the indexed of entries
        # self.uid_table, self.data_table = build_id_key_table(self.nmod,self.ind_tr)
        self.time_data_table_tr = utils.build_time_data_table(self.train_time_ind)
        self.time_data_table_te = utils.build_time_data_table(self.test_time_ind)



    def msg_update_U_llk_single(self,eind_T,T):

        ind_T = self.ind_tr[eind_T]
        y_T = self.y_tr[eind_T].squeeze()
        embed_m = []
        embed_v = []

        start_idx = 0
        for mode, dim in enumerate(self.ndims):

            U_m = (
                self.msg_U_llk_m_del[start_idx + ind_T[mode],:,T]
                .reshape(-1, 1)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            U_v = (
                self.msg_U_llk_v_del[start_idx + ind_T[mode],:,T]
                .reshape(-1, 1)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            embed_m.append(U_m)
            embed_v.append(U_v)

            start_idx = start_idx + dim

        E_z = embed_m[0]
        E_z_2 = torch.diag(embed_v[0]) + torch.mm(E_z, E_z.T)

        for mode in range(1, self.nmod):
            E_u = embed_m[mode]
            E_u_2 = torch.diag(embed_v[mode]) + torch.mm(E_u, E_u.T)

            E_z = E_z * E_u
            E_z_2 = E_z_2 * E_u_2

        E_z = E_z.sum()
        E_z_2 = E_z_2.sum()

        '''fang: try to fix tau now, add update later'''
        E_tau_del = torch.tensor(torch.var(self.y_tr))

        '''fang: standard ADF with single step update'''
        mu = E_z
        sigma = torch.sqrt((1.0 / E_tau_del) + E_z_2 - E_z**2)
        sample = y_T
        dist = torch.distributions.normal.Normal(mu, sigma)
        log_Z = dist.log_prob(sample)
        log_Z.backward()

        start_idx = 0
        for mode, dim in enumerate(self.ndims):
            grad_m = embed_m[mode].grad
            grad_v = embed_v[mode].grad
            m_star = embed_m[mode] + embed_v[mode] * grad_m
            v_star = embed_v[mode] - torch.square(embed_v[mode]) * (
                torch.square(grad_m) - 2 * grad_v
            )

            
            '''' msg update: U_llk =  f_star / f_del (use natural parameters)'''
            msg_U_llk_v_inv_new = 1.0/v_star - 1.0/embed_v[mode]
            msg_U_llk_v_inv_m_new = torch.div(m_star,v_star) - torch.div(embed_m[mode],embed_v[mode])

            # Damping & neg-var check in natural-paras
            msg_U_llk_v_inv_old = (1.0 / self.msg_U_llk_v[start_idx + ind_T[mode],:, T]).reshape(-1, 1)

            msg_U_llk_v_inv_m_old  = torch.div(self.msg_U_llk_m[start_idx + ind_T[mode],:, T], self.msg_U_llk_v[start_idx + ind_T[mode],:, T]).reshape(-1, 1)



            msg_U_llk_v_inv = (
                self.DAMPING * msg_U_llk_v_inv_old
                + (1 - self.DAMPING) * msg_U_llk_v_inv_new
            )
            msg_U_llk_v_inv_m = (
                self.DAMPING
                * msg_U_llk_v_inv_m_old
                + (1 - self.DAMPING) * msg_U_llk_v_inv_m_new
            )

            # transform natrual paras back to mean/var 
            msg_U_llk_v_inv = torch.where(msg_U_llk_v_inv > 0, msg_U_llk_v_inv, 1e0)

            self.msg_U_llk_v[start_idx + ind_T[mode], :, T] = torch.nan_to_num(1.0 / msg_U_llk_v_inv).detach().squeeze()

            self.msg_U_llk_m[start_idx + ind_T[mode], :, T] = torch.nan_to_num(
            (1.0 / msg_U_llk_v_inv) * msg_U_llk_v_inv_m).detach().squeeze()

            start_idx = start_idx + dim


    def 


        

    def msg_update_U_llk(self, T):

        # retrive the observed entries at T
        eind_T_list = self.time_data_table_tr[T]  # id of observed entries at this time-stamp
        

        for eind in eind_T_list:
            self.msg_update_U_llk_single(eind,T)



        # we also update the tau here
        # a = 0.5 * N_T + 1
        # b = (
        #     0.5
        #     * ((y_T * y_T).sum() - 2 * (y_T * E_z_del).sum() + E_z_2_del.sum()).detach()
        # )
        # self.msg_update_tau(a, b, T)

    def arrange_U_llk(self, U_llk_del_T):
        # arrange_U_to_mode-wise for convenience in computing
        U_llk_del_T_m = U_llk_del_T[: self.num_nodes, :]
        U_llk_del_T_v = U_llk_del_T[self.num_nodes :, :]

        # arrange_U_to_mode-wise for convenience in computing
        U_llk_del_T_m = []
        U_llk_del_T_v = []

        idx_start = 0

        for ndim in self.ndims:
            idx_end = idx_start + ndim
            U_llk_del_T_mode_m = U_llk_del_T[idx_start:idx_end, :]
            U_llk_del_T_mode_v = U_llk_del_T[
                self.num_nodes + idx_start : self.num_nodes + idx_end, :
            ]

            U_llk_del_T_m.append(U_llk_del_T_mode_m)
            U_llk_del_T_v.append(U_llk_del_T_mode_v)

            idx_start = idx_end

        return U_llk_del_T_m, U_llk_del_T_v

    def moment_product_U(self, ind_T, U_llk_T_m, U_llk_T_v):
        # compute first and second moments of \Hadmard_prod_{k \in given modes} u_k -CP based on the U_llk_del
        # based on the U_llk_del (calibrating factors)

        E_z = U_llk_T_m[0][ind_T[:, 0]]  # N*R_u
        E_z_expand = E_z.unsqueeze(-1)  # N*R_u*1
        E_z_expand_T = torch.transpose(E_z_expand, dim0=1, dim1=2)  # N*1*R_u
        E_z_2 = torch.diag_embed(U_llk_T_v[0][ind_T[:, 0]], dim1=1) + torch.bmm(
            E_z_expand, E_z_expand_T
        )  # N*R_u*R_u

        for mode in range(1, self.nmod):
            E_u = U_llk_T_m[mode][ind_T[:, mode]]  # N*R_u
            E_u_expand = E_u.unsqueeze(-1)  # N*R_u*1
            E_u_expand_T = torch.transpose(E_u_expand, dim0=1, dim1=2)  # N*1*R_u
            E_u_2 = torch.diag_embed(
                U_llk_T_v[mode][ind_T[:, mode]], dim1=1
            ) + torch.bmm(
                E_u_expand, E_u_expand_T
            )  # N*R_u*R_u

            E_z = E_z * E_u
            E_z_2 = E_z_2 * E_u_2

        # E(1^T z)^2 = trace (1*1^T* z^2)
        if self.R_U > 1:
            return E_z.squeeze().sum(-1), torch.einsum(
                "bii->b",
                torch.matmul(
                    E_z_2, torch.ones(self.R_U, self.R_U).double().to(self.device)
                ),
            )

    def msg_update_tau(self, a, b, T):
        self.msg_tau_a[T] = a
        self.msg_tau_b[T] = b

    def msg_update_tau_del(self, T):
        # add prior db check: done
        self.msg_tau_a_del_T[T] = (
            self.a0
            + self.msg_tau_a[:T].sum()
            + self.msg_tau_a[T + 1 :].sum()
            - self.N_time
            + 1
        )
        self.msg_tau_b_del_T[T] = (
            self.b0 + self.msg_tau_b[:T].sum() + self.msg_tau_b[T + 1 :].sum()
        )

    def msg_update_U_llk_del(self, T):
        # with message-passing framework, we just merge in-var-msg from all branches to get q_del
        # no need to compute from posterior/cur-factor
        # U_llk_del = U_f + U_b

        msg_U_llk_del_v_inv = (
            1.0 / self.msg_U_f_v[: self.num_nodes, :, T]
            + 1.0 / self.msg_U_b_v[: self.num_nodes, :, T]
        )

        msg_U_llk_del_v_inv_m = torch.div(
            self.msg_U_f_m[: self.num_nodes, :, T],
            self.msg_U_f_v[: self.num_nodes, :, T],
        ) + torch.div(
            self.msg_U_b_m[: self.num_nodes, :, T],
            self.msg_U_b_v[: self.num_nodes, :, T],
        )

        self.msg_U_llk_v_del[:, :, T] = torch.nan_to_num(1.0 / msg_U_llk_del_v_inv)
        self.msg_U_llk_m_del[:, :, T] = torch.nan_to_num(
            (1.0 / msg_U_llk_del_v_inv) * msg_U_llk_del_v_inv_m
        )

    def msg_update_U_trans_del(self, T, mode="forward"):
        # U_f_del = U_b + U_llk : msg from var U_T to p(U_T | U_{T-1})
        # (left direction msg, will used during the backward)

        # U_b_del = U_f + U_llk : msg from var U_T to p(U_{T+1} | U_{T})
        # (right direction msg, will used during the forward)

        # double check:done

        # if mode=='forward':
        # for the last time var, we don't have to update its U_b_del (right-out msg)--we'll never use it

        msg_U_b_del_v_inv = (
            1.0 / self.msg_U_f_v[: self.num_nodes, :, T]
            + 1.0 / self.msg_U_llk_v[:, :, T]
        )

        msg_U_b_del_v_inv_m = torch.div(
            self.msg_U_f_m[: self.num_nodes, :, T],
            self.msg_U_f_v[: self.num_nodes, :, T],
        ) + torch.div(self.msg_U_llk_m[:, :, T], self.msg_U_llk_v[:, :, T])

        self.msg_U_b_v_del[: self.num_nodes, :, T] = torch.nan_to_num(
            1.0 / msg_U_b_del_v_inv
        )
        self.msg_U_b_v_del[self.num_nodes :, :, T] = self.msg_U_f_v[
            self.num_nodes :, :, T
        ]

        self.msg_U_b_m_del[: self.num_nodes, :, T] = torch.nan_to_num(
            (1.0 / msg_U_b_del_v_inv) * msg_U_b_del_v_inv_m
        )
        self.msg_U_b_m_del[self.num_nodes :, :, T] = self.msg_U_f_m[
            self.num_nodes :, :, T
        ]

        # else:
        # backward
        # if T>0:
        # for the T0 var, we don't update its U_f_del (left-out msg)--we'll never use it

        msg_U_f_del_v_inv = (
            1.0 / self.msg_U_b_v[: self.num_nodes, :, T]
            + 1.0 / self.msg_U_llk_v[:, :, T]
        )
        msg_U_f_del_v_inv_m = torch.div(
            self.msg_U_b_m[: self.num_nodes, :, T],
            self.msg_U_b_v[: self.num_nodes, :, T],
        ) + torch.div(self.msg_U_llk_m[:, :, T], self.msg_U_llk_v[:, :, T])

        self.msg_U_f_v_del[: self.num_nodes, :, T] = torch.nan_to_num(
            1.0 / msg_U_f_del_v_inv
        )
        self.msg_U_f_v_del[self.num_nodes :, :, T] = self.msg_U_b_v[
            self.num_nodes :, :, T
        ]

        self.msg_U_f_m_del[: self.num_nodes, :, T] = torch.nan_to_num(
            (1.0 / msg_U_f_del_v_inv) * msg_U_f_del_v_inv_m
        )
        self.msg_U_f_m_del[self.num_nodes :, :, T] = self.msg_U_b_m[
            self.num_nodes :, :, T
        ]

    def post_update_U(self):
        # merge all factor->var messages: U_llk, U_f, U_b
        # we only use it for init/ merge-all after training process

        post_U_v_inv_all = (
            1.0 / self.msg_U_f_v[: self.num_nodes, :, :]
            + 1.0 / self.msg_U_b_v[: self.num_nodes, :, :]
            + 1.0 / self.msg_U_llk_v
        )
        post_U_v_inv_m_all = (
            torch.div(
                self.msg_U_f_m[: self.num_nodes, :, :],
                self.msg_U_f_v[: self.num_nodes, :, :],
            )
            + torch.div(
                self.msg_U_b_m[: self.num_nodes, :, :],
                self.msg_U_b_v[: self.num_nodes, :, :],
            )
            + torch.div(self.msg_U_llk_m, self.msg_U_llk_v)
        )

        # arrange the post.U per mode
        self.post_U_m = []
        self.post_U_v = []

        idx_start = 0

        for ndim in self.ndims:
            idx_end = idx_start + ndim
            post_U_mode_v = 1.0 / post_U_v_inv_all[idx_start:idx_end, :, :]
            post_U_mode_m = post_U_mode_v * post_U_v_inv_m_all[idx_start:idx_end, :, :]

            self.post_U_v.append(post_U_mode_v)
            self.post_U_m.append(post_U_mode_m)

            idx_start = idx_end

    def post_update_tau(self):
        self.post_a = self.a0 + self.msg_tau_a.sum() - self.N_time
        self.post_b = self.b0 + self.msg_tau_n.sum()

    def msg_update_U_trans_vec(self, T, mode="forward"):

        time_gap = self.time_uni[T + 1] - self.time_uni[T]
        A_T_block = torch.block_diag(
            *([torch.matrix_exp(self.F * time_gap)] * self.R_U)
        )
        P_inf_block = torch.block_diag(*([self.P_inf] * self.R_U))
        Q_T_block = P_inf_block - P_inf_block @ A_T_block @ P_inf_block.T

        msg_m_l = self.msg_U_b_m_del[:, :, T].T.reshape(-1)
        msg_v_l = self.msg_U_b_v_del[:, :, T].T.reshape(-1)

        # msg from the right (from U_{T+1})
        msg_m_r = self.msg_U_f_m_del[:, :, T + 1].T.reshape(-1).clone()
        msg_v_r = self.msg_U_f_v_del[:, :, T + 1].T.reshape(-1).clone()

        msg_v_l = torch.where(msg_v_l > 0, msg_v_l, 1e5)
        msg_v_r = torch.where(msg_v_r > 0, msg_v_r, 1e5)

        if mode == "forward":
            #  in the forward pass, we only update the msg to right (U_{T+1})
            msg_m_r.requires_grad = True
            msg_v_r.requires_grad = True
            target_m = msg_m_r
            target_v = msg_v_r
        else:
            # in the backward pass, we only update the msg to left (U_{T})
            msg_m_l.requires_grad = True
            msg_v_l.requires_grad = True
            target_m = msg_m_l
            target_v = msg_v_l

        mu = (A_T_block @ msg_m_l.view(-1, 1)).squeeze()  # num_node * 1
        sigma = (
            torch.diag(msg_v_r)
            + Q_T_block
            + A_T_block @ torch.diag(msg_v_l) @ A_T_block.T
        ) - torch.outer(mu, mu)

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
        target_v_star = target_v - torch.square(target_v) * (
            torch.square(target_m_grad) - 2 * target_v_grad
        )

        # target_v_star = torch.where(target_v_star>0,target_v_star,0.5)

        # target_v_star = torch.where(target_v_star>0,target_v_star,1e5)

        # update the factor: msg_new = msg_star / msg_old
        target_v_inv_new = torch.nan_to_num(1.0 / target_v_star - 1.0 / target_v)

        target_v_inv_m_new = torch.nan_to_num(
            torch.div(target_m_star, target_v_star) - torch.div(target_m, target_v)
        )

        if mode == "forward":

            # DAMPING:
            target_v_inv = (
                self.DAMPING * (1.0 / self.msg_U_f_v[:, :, T + 1].T.reshape(-1))
                + (1 - self.DAMPING) * target_v_inv_new
            )

            target_v_inv_m = (
                self.DAMPING
                * torch.div(
                    self.msg_U_f_m[:, :, T + 1], self.msg_U_f_v[:, :, T + 1]
                ).T.reshape(-1)
                + (1 - self.DAMPING) * target_v_inv_m_new
            )

            target_v_inv = torch.where(target_v_inv > 0, target_v_inv, 1e-5)

            self.msg_U_f_v[:, :, T + 1] = (
                (1.0 / target_v_inv).reshape(self.R_U, 2 * self.num_nodes).T
            )
            self.msg_U_f_m[:, :, T + 1] = (
                ((1.0 / target_v_inv) * target_v_inv_m)
                .reshape(self.R_U, 2 * self.num_nodes)
                .T
            )
        else:

            # DAMPING:
            target_v_inv = (
                self.DAMPING * (1.0 / self.msg_U_b_v[:, :, T].T.reshape(-1))
                + (1 - self.DAMPING) * target_v_inv_new
            )

            target_v_inv_m = (
                self.DAMPING
                * torch.div(
                    self.msg_U_b_m[:, :, T], self.msg_U_b_v[:, :, T + 1]
                ).T.reshape(-1)
                + (1 - self.DAMPING) * target_v_inv_m_new
            )

            target_v_inv = torch.where(target_v_inv > 0, target_v_inv, 1e-5)

            self.msg_U_b_v[:, :, T] = (
                (1.0 / target_v_inv).reshape(self.R_U, 2 * self.num_nodes).T
            )
            self.msg_U_b_m[:, :, T] = (
                ((1.0 / target_v_inv) * target_v_inv_m)
                .reshape(self.R_U, 2 * self.num_nodes)
                .T
            )

        # print(T)
        # assert (target_v_star>0).all() == True

    def model_test(self):
        y_pred_list = []

        for tid, T in enumerate(np.unique(self.test_time_ind)):
            eind_T = self.time_data_table_te[
                tid
            ]  # id of observed entries at this time-stamp

            ind_T = self.ind_te[eind_T]

            
            U_post_m_T = [item[:, :,T] for item in self.post_U_m]

            # U_post_v = model.post_U_v[:,:,T]

            E_z = U_post_m_T[0][ind_T[:, 0]]  # N*R_u

            for mode in range(1, self.nmod):
                E_u = U_post_m_T[mode][ind_T[:, mode]]  # N*R_u

                E_z = E_z * E_u

            # print(E_z.sum(-1).shape)
            y_pred_list.append(E_z.sum(-1))

            y_pred = torch.cat(y_pred_list)
        loss = torch.nn.MSELoss()
        rmse = torch.sqrt(loss(y_pred, self.y_te.squeeze()))

        return rmse

    def msg_update_U_trans_EP(self, T, mode="forward"):
        '''applp EP to update the msg of tranfer var'''

        time_gap = self.time_uni[T + 1] - self.time_uni[T]
        A_T = torch.matrix_exp(self.F * time_gap)
        Q_T = self.P_inf - self.P_inf @ A_T @ self.P_inf.T

        for r in range(self.R_U):
            # msg from the left (from U_T)
            # double check: f/b:done

            msg_m_l = self.msg_U_b_m_del[:, r, T]
            msg_v_l = self.msg_U_b_v_del[:, r, T]

            # msg from the right (from U_{T+1})
            msg_m_r = self.msg_U_f_m_del[:, r, T + 1]
            msg_v_r = self.msg_U_f_v_del[:, r, T + 1]

            # msg_v_l = torch.where(msg_v_l > 0, msg_v_l, self.v0.double())
            # msg_v_r = torch.where(msg_v_r > 0, msg_v_r, self.v0.double())

            if mode == "forward":
                #  in the forward pass, we only update the msg to right (U_{T+1})
                self.msg_U_f_m[:, r, T + 1] = (A_T @ msg_m_l.view(-1, 1)).squeeze()
                self.msg_U_f_v[:, r, T + 1] = msg_v_l

                
                
            else:
                # in the backward pass, we only update the msg to left (U_{T})
                
                # self.msg_U_f_m[:, r, T] = torch.linalg.solve(A_T,msg_m_r)
                
                A_T_inv = torch.linalg.inv(A_T)
                self.msg_U_f_m[:, r, T] = (A_T_inv @ msg_m_r.view(-1, 1)).squeeze()

                self.msg_U_f_v[:, r, T] = msg_v_r

            

    def msg_update_U_trans(self, T, mode="forward"):
        

        time_gap = self.time_uni[T + 1] - self.time_uni[T]
        A_T = torch.matrix_exp(self.F * time_gap)
        Q_T = self.P_inf - self.P_inf @ A_T @ self.P_inf.T

        for r in range(self.R_U):
            # msg from the left (from U_T)

            msg_m_l = self.msg_U_b_m_del[:, r, T]
            msg_v_l = self.msg_U_b_v_del[:, r, T]

            # msg from the right (from U_{T+1})
            msg_m_r = self.msg_U_f_m_del[:, r, T + 1]
            msg_v_r = self.msg_U_f_v_del[:, r, T + 1]

            msg_v_l = torch.where(msg_v_l > 0, msg_v_l, self.v0)
            msg_v_r = torch.where(msg_v_r > 0, msg_v_r, self.v0)

            if mode == "forward":
                #  in the forward pass, we only update the msg to right (U_{T+1})
                msg_m_r.requires_grad = True
                msg_v_r.requires_grad = True
                target_m = msg_m_r
                target_v = msg_v_r
            else:
                # in the backward pass, we only update the msg to left (U_{T})
                msg_m_l.requires_grad = True
                msg_v_l.requires_grad = True
                target_m = msg_m_l
                target_v = msg_v_l

            mu = (A_T @ msg_m_l.view(-1, 1)).squeeze()  # num_node * 1
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
            target_v_star = target_v - torch.square(target_v) * (
                torch.square(target_m_grad) - 2 * target_v_grad
            )

            # update the factor: msg_new = msg_star / msg_old
            target_v_inv_new = 1.0 / target_v_star - 1.0 / target_v
            target_v_inv_m_new = torch.div(target_m_star, target_v_star) - torch.div(
                target_m, target_v
            )

            # DOUBLE CHECK:done
            if mode == "forward":
                self.msg_U_f_v[:, r, T + 1] = torch.nan_to_num(1.0 / target_v_inv_new)
                self.msg_U_f_m[:, r, T + 1] = torch.nan_to_num(
                    (1.0 / target_v_inv_new) * target_v_inv_m_new
                )
            else:
                self.msg_U_b_v[:, r, T] = torch.nan_to_num(1.0 / target_v_inv_new)
                self.msg_U_b_m[:, r, T] = torch.nan_to_num(
                    (1.0 / target_v_inv_new) * target_v_inv_m_new
                )



    def msg_update_U_trans_linear(self, T, mode="forward"):

        # time_gap = self.time_uni[T+1] - self.time_uni[T]
        # A_T = torch.matrix_exp(self.F * 0.01)

        # # A_T = torch.eye(2*self.num_nodes).double()

        # A_T_inv = torch.linalg.inv(A_T)
        # Q_T = self.P_inf - self.P_inf @ A_T @ self.P_inf.T

        for r in range(self.R_U):
            # msg from the left (from U_T)
            # double check: f/b:done

            msg_m_l = self.msg_U_b_m_del[:, r, T]
            msg_v_l = self.msg_U_b_v_del[:, r, T]

            # msg from the right (from U_{T+1})
            msg_m_r = self.msg_U_f_m_del[:, r, T + 1]
            msg_v_r = self.msg_U_f_v_del[:, r, T + 1]

            # DOUBLE CHECK:done
            if mode == "forward":

                self.msg_U_f_v[:, r, T + 1] = msg_v_l
                self.msg_U_f_m[:, r, T + 1] = msg_m_l
            else:
                self.msg_U_b_v[:, r, T] = msg_v_r
                self.msg_U_b_m[:, r, T] = msg_m_r
