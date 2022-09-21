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

To-do-list:
- single ADF on the main model! - 9.21    
- debug the ADF_update_T
- double-check the update of \tau and sigma to compute log_Z 
- check the ADF impletation on Beijing
- check the CEP impletation on Beijing
- implement the log function and yaml-params functions
- unify the tensor-series-data class