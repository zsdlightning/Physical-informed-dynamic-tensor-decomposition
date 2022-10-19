import numpy as numpy
import torch 
from model_dynamic_CP import LDS_dynammic_CP
import utils
import tqdm


# load config
args = None

# load data 

# model 
model = LDS_dynammic_CP()


# psudo code to update 
for i in tqdm.tqdm(range(args.epoch)):

    for mode in range(model.nmod):
        for k in range(model.N_time):

            # update model.msg_U_M, model.msg_U_V : compute Ez, Ez2, get \beta, S, emseble to long vec msg_U
            model.msg_update_U(mode,k)

            # Kalman Filter: predict and update 
            model.LDS_list[mode].filter_predict(ind=k)
            model.LDS_list[mode].filter_update(y= model.msg_U_M, R = model.msg_U_V)
        
        model.LDS_list[mode].smooth()

        model.post_update_U(mode)

    test_result = model.test()

utils.make_log(test_result)

