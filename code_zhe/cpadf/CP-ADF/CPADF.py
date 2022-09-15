import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time

torch.manual_seed(0)
np.random.seed(0)

class CPADF:
    def __init__(self, nvec, R):
        self.nmod = len(nvec)
        self.R = R
        self.tau = 1.0
        #initialize with  prior 
        self.mu = [torch.tensor(np.random.rand(nvec[j], self.R), dtype=torch.float32, requires_grad=False) for j in range(self.nmod)]
        self.v = [torch.tensor(np.ones([nvec[j], self.R]), dtype=torch.float32, requires_grad=False) for j in range(self.nmod)]

    #input is a vector
    def get_logz(self, mu, v, y):
        #nmod x R
        mu = mu.view(self.nmod, -1) 
        v = v.view(self.nmod, -1)
        f_mean = mu[0,:]
        mm2 = torch.outer(mu[0,:], mu[0,:]) + torch.diag(v[0,:])
        for j in range(1, self.nmod):
            mm2 = mm2*(torch.outer(mu[j,:], mu[j,:]) + torch.diag(v[j,:]))
            f_mean = f_mean*mu[j,:]
        f_mean = torch.sum(f_mean)
        f_v =  torch.sum(mm2) - f_mean**2
        y_v = f_v + self.tau
        logZ = -0.5*torch.log(torch.tensor(2*np.pi)) - 0.5*torch.log(y_v) - 0.5/y_v*(y - f_mean)**2
        return logZ


    def go_through(self, ind, y, test_ind, test_y):
        #let fix tau for convenience 
        self.tau = torch.tensor(np.var(y))
        N = ind.shape[0]
        for npass in range(10):
            for n in range(N):
                ind_n = ind[n,:]
                y_n = y[n]
                #nmod * R X 1
                mu_n = torch.hstack([self.mu[j][ ind_n[j] ] for j in range(self.nmod)]).clone().detach().requires_grad_(True)
                v_n = torch.hstack([self.v[j][ ind_n[j] ] for j in range(self.nmod)]).clone().detach().requires_grad_(True)
                logZ = self.get_logz(mu_n, v_n, y_n)
                dmu = torch.autograd.grad(logZ, mu_n, create_graph=True)[0]
                dv = torch.autograd.grad(logZ, v_n, create_graph=True)[0]
                mu_star = mu_n + v_n*dmu
                v_star = v_n - (v_n**2)*(dmu**2 -2*dv)
                mu_star = mu_star.view(self.nmod, -1)
                v_star = v_star.view(self.nmod, -1)
                for j in range(self.nmod):
                    self.mu[j][ ind_n[j] ] = mu_star[j,:].clone().detach()
                    self.v[j][ ind_n[j] ] = v_star[j,:].clone().detach()
                if n%100 == 0 or n==N-1:
                    pred = self.pred(test_ind, test_y)
                    rmse = torch.sqrt(torch.mean( torch.square(pred - test_y) ))
                    print('pass #%d, %d points, rmse = %g'%(npass, n+1, rmse))
                    

    def pred(self, test_ind, test_y):
        pred = self.mu[0][test_ind[:,0]]
        for j in range(1,self.nmod):
            pred = pred*self.mu[j][test_ind[:,j]]
        pred = torch.sum(pred, 1)
        return pred 

def test_beijing():
    ind = []
    y = []
    with open('./data/beijing_15k_train.txt','r') as f:
            for line in f:
                items = line.strip().split(',')
                y.append(float(items[-1]))
                ind.append([int(idx) for idx in items[0:-1]])
            ind = np.array(ind)
            y = np.array(y)
    nvec = np.max(ind, 0) + 1

    test_ind = []
    test_y = []
    with open('./data/beijing_15k_test.txt','r') as f:
                for line in f:
                    items = line.strip().split(',')
                    test_y.append(float(items[-1]))
                    test_ind.append([int(idx) for idx in items[0:-1]])
                test_ind = np.array(test_ind)
                test_y = np.array(test_y)

    R = 2
    model = CPADF(nvec, R)
    model.go_through(ind, y, test_ind, test_y)

if __name__ == '__main__':
    test_beijing()

