import numpy as np
import torch
import utils
import scipy
from scipy import linalg
from torch.utils.data import Dataset


def build_time_data_table(time_ind):
    # input: sorted time-stamp seq (duplicated items exists) attached with data seq
    # output: table (list) of associated data points of each timestamp
    # ref: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function/43094244
    # attention, here the input "time-stamps" can be either (repeating) id, or exact values, but seq length must match data seq
    # in out table, order of item represents the time id in order
    time_data_table = np.split(
        np.array([i for i in range(len(time_ind))]),
        np.unique(time_ind, return_index=True)[1][1:],
    )
    return time_data_table


def build_id_key_table(nmod, ind):
    # build uid-data_key_table, implement by nested list

    # store the indices of associated nodes in each mode over all obseved entries
    uid_table = []

    # store the indices of obseved entries for each node of each mode
    data_table = []

    for i in range(nmod):
        values, inv_id = np.unique(ind[:, i], return_inverse=True)
        uid_table.append(list(values))

        sub_data_table = []
        for j in range(len(values)):
            data_id = np.argwhere(inv_id == j)
            if len(data_id) > 1:
                data_id = data_id.squeeze().tolist()
            else:
                data_id = [[data_id.squeeze().tolist()]]
            sub_data_table.append(data_id)

        data_table.append(sub_data_table)

    return uid_table, data_table


def generate_mask(ndims, ind):
    num_node = sum(ndims)
    nmod = len(ndims)
    ind = torch.tensor(ind)

    mask = torch.zeros((num_node, num_node))
    for i in range(1, nmod):
        row = np.sum(ndims[:i])
        for j in range(i):
            col = np.sum(ndims[:j])
            indij = ind[:, [i, j]]
            indij = torch.unique(indij, dim=0).long()
            row_idx = row + indij[:, 0]
            col_idx = col + indij[:, 1]
            mask[row_idx.long(), col_idx.long()] = 1
    return mask


def generate_Lapla(ndims, ind):
    """
    generate the fixed Laplacian mat of prior K-partition graph,
    which is defined by the observed entries in training set
    """
    num_node = sum(ndims)

    W_init = torch.ones((num_node, num_node))
    mask = generate_mask(ndims, ind)
    Wtril = torch.tril(W_init) * mask
    W = Wtril + Wtril.T
    D = torch.diag(W.sum(1))
    return W - D


def generate_state_space_Matern_23(data_dict, hyper_dict):
    """
    For matern 3/2 kernel with given hyper-paras and data,
    generate the parameters of coorspoding state_space_model,
    recall: for each dim of all-node-embedding, the form of state_space_model is iid (independent & identical)

    input: data_dict, hyper_dict
    output: trans mat: F,  stationary covarianc: P_inf

    """

    ndims = data_dict["ndims"]
    D = sum(ndims)
    ind = data_dict["tr_ind"]

    # hyper-para of kernel
    lengthscale = hyper_dict["ls"]
    variance = hyper_dict["var"]
    c = hyper_dict["c"]  # diffusion rate

    lamb = np.sqrt(3) / lengthscale

    # F = torch.zeros((2*D, 2*D), device=data_dict['device'])
    F = np.zeros((2 * D, 2 * D))
    F[:D, :D] = utils.generate_Lapla(ndims, ind) * c
    F[:D, D:] = np.eye(D)
    F[D:, :D] = -np.square(lamb) * np.eye(D)
    F[D:, D:] = -2 * lamb * np.eye(D)

    Q_c = 4 * lamb**3 * variance * np.eye(D)
    L = np.zeros((2 * D, D))
    L[D:, :] = np.eye(D)
    Q = -np.matmul(np.matmul(L, Q_c), L.T)

    P_inf = Lyapunov_slover(F, Q)

    return torch.tensor(F, device=hyper_dict["device"]), torch.tensor(
        P_inf, device=hyper_dict["device"]
    )


def Lyapunov_slover(F, Q):
    """
    For the given mix-process SDE, solve correspoding Lyapunov to get P_{\inf}
    """

    return linalg.solve_continuous_lyapunov(F, Q)


def nan_check_1(model, T):
    msg_list = [
        model.msg_U_llk_m[:, :, T],
        model.msg_U_llk_v[:, :, T],
        model.msg_U_f_m[:, :, T],
        model.msg_U_f_v[:, :, T],
        model.msg_U_b_m[:, :, T],
        model.msg_U_b_v[:, :, T],
        model.msg_U_llk_m_del[:, :, T],
        model.msg_U_llk_v_del[:, :, T],
        model.msg_U_f_m_del[:, :, T],
        model.msg_U_f_v_del[:, :, T],
        model.msg_U_b_m_del[:, :, T],
        model.msg_U_b_v_del[:, :, T],
    ]

    msg_name_list = [
        "msg_U_llk_m",
        "msg_U_llk_v",
        "msg_U_f_m",
        "msg_U_f_v",
        "msg_U_b_m",
        "msg_U_b_v",
        "msg_U_llk_m_del",
        "msg_U_llk_v_del",
        "msg_U_f_m_del",
        "msg_U_f_v_del",
        "msg_U_b_m_del",
        "msg_U_b_v_del",
    ]
    for id, msg in enumerate(msg_list):
        if msg.isnan().any():
            print("invalid number: %s at time %d " % (msg_name_list[id], T))
            return False

    return True


def neg_check_v(model, T):
    msg_list = [
        model.msg_U_llk_v[:, :, T],
        model.msg_U_f_v[:, :, T],
        model.msg_U_b_v[:, :, T],
        model.msg_U_llk_v_del[:, :, T],
        model.msg_U_f_v_del[:, :, T],
        model.msg_U_b_v_del[:, :, T],
    ]

    msg_name_list = [
        "msg_U_llk_v",
        "msg_U_f_v",
        "msg_U_b_v",
        "msg_U_llk_v_del",
        "msg_U_f_v_del",
        "msg_U_b_v_del",
    ]

    for id, msg in enumerate(msg_list):
        if (msg <= 0).any():
            print("invalid v: %s at time %d " % (msg_name_list[id], T))

            return False

    return True


# batch knorker product
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum("bac,bkp->bakcp", A, B).view(
        A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2)
    )
    return res


def Hadamard_product_batch(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Hadamard Products
    :param A: has shape (N, a, b)
    :param B: has shape (N, a, b)
    :return: (N, a, b)
    """
    assert A.dim() == 3 and B.dim() == 3
    assert A.shape == B.shape
    res = A * B
    return res


# batch knorker product
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum("bac,bkp->bakcp", A, B).view(
        A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2)
    )
    return res


def moment_Hadmard(
    modes, ind, U_m, U_v, order="first", sum_2_scaler=True, device=torch.device("cpu")
):
    """
    -compute first and second moments of \Hadmard_prod_{k \in given modes} u_k -CP style
    -can be used to compute full-mode / calibrating-mode of U/gamma ?

    :param modes: list of target mode
    :param ind: index of tensor entries     : shape (N, nmod)
    :param U_m: mean of U                   : shape (nmod,R_U,1)
    :param U_v: var of U (diag)             : shape (nmod,R_U,1) or (nmod,R_U,R_U)
    :param order: oder of expectated order  : "first" or "second"
    :param sum_2_scaler: flag on whether sum the moment 2 scaler  : Bool

    retrun:
    --if sum_2_scaler is True
    : E_z: first moment of 1^T (\Hadmard_prod)  : shape (N, 1)
    : E_z_2: second moment 1^T (\Hadmard_prod)  : shape (N, 1)

    --if sum_2_scaler is False
    : E_z: first moment of \Hadmard_prod   : shape (N, R_U, 1)
    : E_z_2: second moment of \Hadmard_prod: shape (N, R_U, R_U)

    it's easy to transfer this function to kronecker_product(Tucker form) by changing Hadmard_product_batch to kronecker_product_einsum_batched

    """
    assert order in {"first", "second"}
    assert sum_2_scaler in {True, False}

    last_mode = modes[-1]

    diag_cov = True if U_v.size()[-1] == 1 else False

    R_U = U_v.size()[1]

    if order == "first":
        # only compute the first order moment

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        for mode in reversed(modes[:-1]):
            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1
            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1

        return E_z.sum(dim=1) if sum_2_scaler else E_z

    elif order == "second":
        # compute the second order moment E_z / E_z_2

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        if diag_cov:
            # diagnal cov
            E_z_2 = torch.diag_embed(
                U_v[last_mode][ind[:, last_mode]], dim1=1
            ) + torch.bmm(
                E_z, E_z.transpose(dim0=1, dim1=2)
            )  # N*R_u*R_U

        else:
            # full cov
            E_z_2 = U_v[last_mode][ind[:, last_mode]] + torch.bmm(
                E_z, E_z.transpose(dim0=1, dim1=2)
            )  # N*R_u*R_U

        for mode in reversed(modes[:-1]):

            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1

            if diag_cov:

                E_u_2 = torch.diag_embed(
                    U_v[last_mode][ind[:, last_mode]], dim1=1
                ) + torch.bmm(
                    E_u, E_u.transpose(dim0=1, dim1=2)
                )  # N*R_u*R_U

            else:
                E_u_2 = U_v[last_mode][ind[:, last_mode]] + torch.bmm(
                    E_u, E_u.transpose(dim0=1, dim1=2)
                )  # N*R_u*R_U

            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1
            E_z_2 = Hadamard_product_batch(E_z_2, E_u_2)  # N*R_u*R_u

        if sum_2_scaler:
            E_z = E_z.sum(dim=1)  # N*R_u*1 -> N*1

            # E(1^T z)^2 = trace (1*1^T* z^2)

            E_z_2 = torch.einsum(
                "bii->b", torch.matmul(E_z_2, torch.ones(R_U, R_U).to(device))
            ).unsqueeze(
                -1
            )  # N*R_u*R_u -> -> N*1

        return E_z, E_z_2
