# Streaming Factor Trajectory Learning for Dynamic Tensor Decomposition


we gonna try the new simple idea, streaming dynamic CP. saying: 

Y(t) = CP( U1(t),U2(t)... )

just set independent temporal-GP/SDE/LDSs over U1(t), U2(t)... seperately

inference is achieved by  Bayesian filter & smooth

draft link: https://www.overleaf.com/project/6363a960485a46499baef800

To-do-list:
-finish dynamic-streaming-CP main function (what's the logics): done
-finish the class: model-dynamic-streaming-CP
  - model.track_envloved_objects:done
  - model.filter_predict: done
  - model.filter_update: done
  - model.smooth:done
  - model.msg_approx:done
  - model.get_post_U:done
  - model.test:done
  - tau_update:to be in the next interation
-some new utils function
-make it runable
-alternative test/get_post_U:
  - not load the post_U_m for each objects 
  - go through all test samples, only touch the smoothed states of the involved object traj 



Unit-Test-list :
- functions
  - 

Optimize-list :
- 

Observation:
