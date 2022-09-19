# Physical-informed-dynamic-tensor-decomposition


Current bugs:

-On Beijing-dataset, test rmse drop to 0.95 at forward-1-epoch, remian the level for several epochs, then divergence 

-the my'ADF on beijing(static_ADF.py) still not work well, but shandian's code(CPADF.py,CPD.py ) works well

To-do-list:
- check the ADF impletation on Beijing
- check the CEP impletation on Beijing
- implement the log function and yaml-params functions
- unify the tensor-series-data class