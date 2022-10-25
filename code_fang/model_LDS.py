import numpy as np
# import scipy
import torch
import utils

'''
base model of LDS system

SDE represent of a (1-d) GP with stationary kernel :

dx/dt = Fx(t) + Lw(t)

the coorespoding LDS model is :

transition: x_k = A_{k-1} * x_{k-1} + q_{k-1}
enission: y_k = H*x_k + noise(R)

where: A_{k-1} = mat_exp(F*(t_k-t_{k-1})), q_{k-1}~N(0,P_{\inf}-A_{k-1}P_{\inf}A_{k-1}^T)

Attention, with Matern /nu=1, x is 2-d vector = (f, df/dt), H = (1,0)

'''
# SDE-inference
class LDS_GP():
    def __init__(self,hyper_para_dict):
        self.device = hyper_para_dict['device'] # add the cuda version later 

        # self.N = hyper_para_dict['N'] # number of data-llk
        self.N_time = hyper_para_dict['N_time'] # number of total states

        self.F = hyper_para_dict['F'].double().to(self.device) # transition mat-SDE
        self.H = hyper_para_dict['H'].double().to(self.device) # emission mat-SDE
        self.R = hyper_para_dict['R'].double().to(self.device) # emission noise

        self.P_inf = hyper_para_dict['P_inf'].double().to(self.device) # P_inf 
        
        self.fix_int = hyper_para_dict['fix_int'] # whether the time interval is fixed
        
        if self.fix_int:
            self.A = torch.matrix_exp(self.F*self.fix_int)
            self.Q = self.P_inf - torch.mm(torch.mm(self.A,self.P_inf),self.A.T)
        else:
            # pre-compute and store the transition-mats for dynamic gap
            # we can also do this in each filter predict-step, but relative slow 
            '''check whether compute A,Q here or at  predict-step'''

            self.time_int_list = hyper_para_dict['time_int_list'].to(self.device)

            # self.A_list = [torch.matrix_exp(self.F*time_int).double() for time_int in self.time_int_list]
            # self.Q_list = [self.P_inf - torch.mm(torch.mm(A,self.P_inf),A.T) for A in self.A_list]

        self.m_0 = hyper_para_dict['m_0'].double().to(self.device) # init mean
        self.P_0 = hyper_para_dict['P_0'].double().to(self.device) # init var

        self.reset_list()


    def reset_list(self):
        self.m = self.m_0 # store the current state(mean)
        self.P = self.P_0 # store the current state(var)

        self.m_update_list = [] # store the filter-update state(mean)
        self.P_update_list = [] # store the filter-update state(var)

        self.m_pred_list = [] # store the filter-pred state(mean)
        self.P_pred_list = [] # store the filter-pred state(var)

        self.m_smooth_list = [] # store the smoothed state(mean)
        self.P_smooth_list = [] # store the smoothed state(mean)
 
    def filter_predict(self,ind=None,time_int=None):
        if self.fix_int:
            self.m = torch.mm(self.A,self.m).double()
            self.P = torch.mm(torch.mm(self.A,self.P),self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

        # none-fix-interval, recompute A,Q based on current time-interval
        else:

            if ind is None:
                raise Exception('need to input the state-index for non-fix-interval case')

            time_int = self.time_int_list[ind]
            self.A = torch.matrix_exp(self.F*time_int).double()
            self.Q = self.P_inf - torch.mm(torch.mm(self.A,self.P_inf),self.A.T)

            # self.A = self.A_list[ind]
            # self.Q = self.Q_list[ind]
            
            self.m = torch.mm(self.A,self.m).double()
            self.P = torch.mm(torch.mm(self.A,self.P),self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

            # self.tau_list.append(tau) # store all the time interval

    def filter_update(self,y,R=None):

        if R is None:
            R = self.R
        
        V = y-torch.mm(self.H,self.m)
        S = torch.mm(torch.mm(self.H,self.P),self.H.T) + R
        K = torch.mm(torch.mm(self.P,self.H.T),torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K,V)
        self.P = self.P - torch.mm(torch.mm(K,self.H),self.P)

        self.m_update_list.append(self.m)
        self.P_update_list.append(self.P)

    def smooth(self):
        
        # start from the last end
        m_s = self.m_update_list[-1]
        P_s = self.P_update_list[-1]

        self.m_smooth_list.insert(0,m_s)
        self.P_smooth_list.insert(0,P_s)

        if self.fix_int:
            for i in reversed(range(self.N_time-1)):

                m = self.m_update_list[i]
                P = self.P_update_list[i]

                m_pred = self.m_pred_list[i+1]
                P_pred = self.P_pred_list[i+1]

                G = torch.mm(torch.mm(P,self.A.T),torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G,m_s-m_pred)
                P_s = P + torch.mm(torch.mm(G,P_s-P_pred),G.T)
            
                self.m_smooth_list.insert(0,m_s)
                self.P_smooth_list.insert(0,P_s)

        else: # to be verify
            for i in reversed(range(self.N_time-1)):

                m = self.m_update_list[i]
                P = self.P_update_list[i]

                m_pred = self.m_pred_list[i+1]
                P_pred = self.P_pred_list[i+1]

                
                time_int = self.time_int_list[i+1]
                A = torch.matrix_exp(self.F*time_int)
                # A = self.A_list[i+1]

                G = torch.mm(torch.mm(P,A.T),torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G,m_s-m_pred)
                P_s = P + torch.mm(torch.mm(G,P_s-P_pred),G.T)
            
                self.m_smooth_list.insert(0,m_s)
                self.P_smooth_list.insert(0,P_s)
                

