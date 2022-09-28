# Physical-informed-dynamic-tensor-decomposition


Current bugs:

-On Beijing-dataset, test rmse drop to 0.95 at forward-1-epoch, remian the level for several epochs, then divergence 

-the my'ADF on beijing(static_ADF_beijing.py) still not work well, but shandian's code(CPADF.py,CPD.py ) works well: ADF_update_N
    -- fixed, stupid typo (forget to update start_idx in the for loop) 
    -- double-check the update of \tau and sigma to compute log_Z in main model


Observation:
- for static-ADF-ADF_update_N, fis tau = var(y) seems make it more stable , easy & faster to converge
- static-ADF_update_T not work -> problem of batch ADF -> if sloved, the whole project is promising 
- Batch-ADF seems to never work...let's try single ADF on the main model! - 9.21
- Linear transfer + single-data-ADF-update on llk msg seems work, but performence still a little worse then pure ADF - 9.24
- the results are sensitive to self.v0, 1e2 seems the best now - 9.24 
- Try EP update on the transfer update  -> still not work

To-do-list:
- try the decompose form of CP, still with the mssage passing framework  
- single ADF on the main model! - 9.21: done     
- debug the ADF_update_T
- double-check the update of \tau and sigma to compute log_Z 
- check the ADF impletation on Beijing
- check the CEP impletation on Beijing
- implement the log function and yaml-params functions
- unify the tensor-series-data class

To-do-list-"decomp-CP-idea":
- finish the key msg_update_functions 
    - finish the ultis- moment_Hadmard, the unified func to compute the moments of U/Gamma
- excution order of msg_update_llk_2_U(), msg_update_llk_2_Gamma(), post_update_U(), post_update_Gamma()
    - follow the BCTT? parallel phase: msg_update_llk_2_U()->post_update_U() || seq phase: msg_update_llk_2_Gamma(T)
- model_test function
- key ulti func: moment_product_U, moment_product_Gamma -> how to use them to get parallel CEP update    