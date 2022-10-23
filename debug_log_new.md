# Physical-informed-dynamic-tensor-decomposition


we gonna try the new simple idea, dynamic CP. saying: 

Y(t) = CP( U1(t),U2(t)... )

just set independent temporal-GP/SDE/LDSs over U1(t), U2(t)... seperately

inference is achieved by alternatively apply Bayesian filter & smooth)



To-do-list:
-data and config loading 
-debug
-test 


Unit-Test-list of "decomp-CP":
- functions
  - ultis:moment_Hadmard, moment_Hadmard_T-doing!!
  - model:msg_update_llk_2_U()-doing!!

Optimize-list of "decomp-CP":
- model:msg_update_llk_2_U()
    - put utils.moment_Hadmard_T and utils.moment_Hadmard outside the loop -> faster? 
