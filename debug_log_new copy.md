# Streaming Factor Trajectory Learning for Dynamic Tensor Decomposition


we gonna try the new simple idea, dynamic CP. saying: 

Y(t) = CP( U1(t),U2(t)... )

just set independent temporal-GP/SDE/LDSs over U1(t), U2(t)... seperately

inference is achieved by  Bayesian filter & smooth)

draft link: https://www.overleaf.com/project/6363a960485a46499baef800

To-do-list:
-finish the class: model-dynamic-streaming-CP
  - the msg_update
  - the post_update
-finish dynamic-streaming-CP main function (what's the logics)
-some new utils function
-make it runable

Unit-Test-list :
- functions
  - ultis:moment_Hadmard, moment_Hadmard_T-doing!!
  - model:msg_update_llk_2_U()-doing!!

Optimize-list :
- model:msg_update_llk_2_U()
    - put utils.moment_Hadmard_T and utils.moment_Hadmard outside the loop -> faster? 

Observation:
