import numpy as numpy
import torch
from model_dynamic_streaming_CP import LDS_dynammic_streaming
import utils_streaming
import tqdm
import yaml


# load config
args = None
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

data_path = "../processed_data/beijing_15k.npy"

# prepare hyper_dict and data_dict

hyper_dict = utils_streaming.make_hyper_dict(config, args)
data_dict = utils_streaming.make_data_dict(config, args, hyper_dict)

# model
model = LDS_dynammic_streaming(hyper_dict, data_dict)


for T in range(model.N_time):

    # track the id of envloved objects, will be used in next steps
    model.track_envloved_objects(T)

    # KF prediction step: trajectories of involved objects take Gaussian Jump + update the posterior
    model.filter_predict(T)

    # approx the msg from the group of data-llk at T
    model.msg_approx(T)

    # KF update step: merge prediction-states of KF & data-llk-msg + update the posterior
    model.filter_update(T)

    test_result = model.test()

# RTS-smooth-step
model.smooth()

# get the final posterior of U using the smooth-state of LDS
model.get_final_U()

test_result = model.test()
utils_streaming.make_log(test_result)
