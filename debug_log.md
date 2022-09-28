# Physical-informed-dynamic-tensor-decomposition


Current bugs:

-On Beijing-dataset, test rmse drop to 0.95 at forward-1-epoch, remian the level for several epochs, then divergence 

-the my'ADF on beijing(static_ADF_beijing.py) still not work well, but shandian's code(CPADF.py,CPD.py ) works well: ADF_update_N
    -- fixed, stupid typo (forget to update start_idx in the for loop) 
    -- double-check the update of \tau and sigma to compute log_Z in main model

-Batch-ADF (multi-data-llk per time) seems to never work w/o the msg-passing framework


Observation:
- for static-ADF-ADF_update_N, fis tau = var(y) seems make it more stable , easy & faster to converge
- static-ADF_update_T not work -> problem of batch ADF -> always with invalid cov
- Batch-ADF seems to never work...let's try single ADF on the main model! - 9.21
- Linear transfer + single-data-ADF-update on llk msg seems work, but performence(0.86) still a little worse then pure single ADF(0.85) - 9.24
  
- the results are sensitive to self.v0, 1e2 seems the best now - 9.24 
  
- Try single ADF EP update on the transition update  -> still not work - 9.25

To-do-list:
- try the decompose form of CP, still with the mssage passing framework :doing!!  
- single ADF on the main model! - 9.21: done     
- debug the ADF_update_T: done
- double-check the update of \tau and sigma to compute log_ZL done
- check the ADF impletation on Beijing :done
- check the CEP impletation on Beijing 
- implement the log function and yaml-params functions
- unify the tensor-series-data class

To-do-list-"decomp-CP-idea":
- finish the key msg_update_functions 
    - ultis: moment_Hadmard, moment_Hadmard_T -finish- 9.27
    - model: msg_update_llk_2_U() -finish- 9.28


- excution order of msg_update_llk_2_U(), msg_update_llk_2_Gamma(), post_update_U(), post_update_Gamma()
    - try to follow the BCTT
        -  parallel phase: msg_update_llk_2_U()->post_update_U() 
        -  seq phase: msg_update_llk_2_Gamma(T)
  
- model_test function
- draft 
- implement the log function and yaml-params functions
- key ulti func: moment_product_U, moment_product_Gamma -> how to use them to get parallel CEP update:done    


Unit-Test-list of "decomp-CP":
- functions
  - ultis:moment_Hadmard, moment_Hadmard_T-doing!!
  - model:msg_update_llk_2_U()-doing!!

Optimize-list of "decomp-CP":
- model:msg_update_llk_2_U()
    - put utils.moment_Hadmard_T and utils.moment_Hadmard outside the loop -> faster? 
