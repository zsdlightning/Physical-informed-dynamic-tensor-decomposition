import numpy as numpy
import torch
from model_dynamic_CP import LDS_dynammic_CP
import utils
import tqdm
import yaml


# load config
args = None
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

data_path = "../processed_data/beijing_15k.npy"

# prepare hyper_dict and data_dict

hyper_dict = utils.make_hyper_dict(config, args)
data_dict = utils.make_data_dict(config, args, hyper_dict)

# model
model = LDS_dynammic_CP(hyper_dict, data_dict)


# psudo code to update
for i in tqdm.tqdm(range(args.epoch)):

    for mode in range(model.nmods):

        model.LDS_list[mode].reset_list()

        for k in range(model.N_time):

            # update model.msg_U_M, model.msg_U_V : compute Ez, Ez2, get \beta, S, emseble to long vec msg_U
            model.msg_update_U(mode, k)

            # Kalman Filter: predict and update
            model.LDS_list[mode].filter_predict(ind=k)
            model.LDS_list[mode].filter_update(y=model.msg_U_M, R=model.msg_U_V)

        model.LDS_list[mode].smooth()

        model.post_update_U(mode)

    test_result = model.test()

utils.make_log(test_result)
