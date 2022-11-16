# Streaming Factor Trajectory Learning for Dynamic Tensor Decomposition


we gonna try the new simple idea, streaming dynamic CP. saying: 

Y(t) = CP( U1(t),U2(t)... )

just set independent temporal-GP/SDE/LDSs over U1(t), U2(t)... seperately

inference is achieved by  Bayesian filter & smooth

draft link: https://www.overleaf.com/project/6363a960485a46499baef800

To-do-list:
-finish the class: model-dynamic-streaming-CP
  - model.track_envloved_objects:done
  - model.filter_predict: doing!!
  - model.filter_update
  - model.smooth
  - model.msg_approx:doing!!
  - model.get_post_U
  - tau_update:to be in the next interation
-finish dynamic-streaming-CP main function (what's the logics): done
-some new utils function
-make it runable

Unit-Test-list :
- functions
  - 

Optimize-list :
- 

Observation:
