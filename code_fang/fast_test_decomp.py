import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy

# import pandas
import torch
import utils
from utils import generate_state_space_Matern_23
from scipy import linalg
from utils import build_id_key_table
from model_Bayes_diffusion_decomp import Bayes_diffu_tensor_decomp as Bayes_diffu_tensor
import tqdm

torch.manual_seed(123)

data_file = "../processed_data/beijing_15k.npy"
# data_file = '../processed_data/ctr_10k.npy'

# data_file = '../processed_data/mvlens_10k.npy'
# data_file = '../processed_data/server_10k.npy'
# data_file = '../processed_data/dblp_50k.npy'


full_data = np.load(data_file, allow_pickle=True).item()

fold = 0

# here should add one more data-loader class
data_dict = full_data["data"][fold]
data_dict["ndims"] = full_data["ndims"]
data_dict["num_node"] = full_data["num_node"]

data_dict["time_id_table"] = full_data["time_id_table"]
data_dict["time_uni"] = full_data["time_uni"]

# print(full_data['time_uni']*100)
hyper_dict = {}

hyper_dict["epoch"] = 5
hyper_dict["ls"] = 1
hyper_dict["var"] = 0.1

# hyper_dict['device'] = torch.device("cuda")
hyper_dict["device"] = torch.device("cpu")  # CPU IS MUCH FASTER


hyper_dict["R_U"] = 2  # dim of each node embedding
hyper_dict["c"] = 10  # diffusion rate
hyper_dict["a0"] = 1.0
hyper_dict["b0"] = 1.0
hyper_dict["DAMPING"] = 0.8


F, P_inf = utils.generate_state_space_Matern_23(data_dict, hyper_dict)

data_dict["F"] = F
data_dict["P_inf"] = P_inf

N_T = len(data_dict["time_uni"])

model = Bayes_diffu_tensor(data_dict, hyper_dict)

test_rmse = model.model_test()
print("init state: test_rmse: %.4f " % (test_rmse))

EPOCH = 50
for epoch in tqdm.tqdm(range(EPOCH)):

    # CEP update on U (static Embedding)

    """include update of msg_U and msg_llk_2_Gamma"""
    # model.msg_update_llk()

    model.msg_update_llk_2_U()

    # forward-backward message-passing update on msg_Gamma

    # forward
    for T in range(N_T - 1):

        model.msg_update_llk_2_Gamma(T)  # at N_T - 1?

        model.msg_update_Gamma_f_Trans(T)
        model.msg_update_Trans_f_Gamma(T)

    model.msg_update_Gamma_2_llk(N_T - 1)

    model.post_update_Gamma()
    test_rmse = model.model_test()
    print("it-half: %d, test_rmse: %.4f " % (epoch, test_rmse))

    # backward
    for T in reversed(range(N_T - 1)):
        model.msg_update_Gamma_b_Trans(T)
        model.msg_update_Trans_b_Gamma(T)
        model.msg_update_Gamma_2_llk(T)

    model.post_update_U()
    model.post_update_Gamma()

    test_rmse = model.model_test()
    print("it %d, test_rmse: %.4f " % (epoch, test_rmse))
