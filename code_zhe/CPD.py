import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io

# from scipy.interpolate import griddata
# from pyDOE import lhs
import time

torch.manual_seed(0)
np.random.seed(0)


class CPD:
    # ns: number of time steps
    def __init__(self, nvec, R, ns, max_epoch):
        self.nmod = len(nvec)
        self.nepoch = max_epoch
        self.R = R
        self.d = 0
        self.ns = ns
        self.nvec = nvec
        for j in range(self.nmod):
            self.d = self.d + self.nvec[j]
        self.U = torch.tensor(
            np.random.rand(self.ns, self.d, self.R),
            dtype=torch.float32,
            requires_grad=True,
        )
        self.log_tau = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.log_v = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    def get_loss(self, train_ind, train_y):
        ll = (
            -0.5 * torch.sum(torch.square(self.U[0]))
            - 0.5 * self.d * self.R * (self.ns - 1) * self.log_v
        )
        ll = ll - 0.5 * torch.exp(-self.log_v) * torch.sum(
            torch.square(self.U[1:] - self.U[0:-1])
        )
        Ntr = train_ind.shape[0]
        ll = ll - 0.5 * Ntr * torch.exp(self.log_tau)
        # T x d x R
        U = torch.split(self.U, self.nvec.tolist(), dim=1)
        # U0: T x d_1 x R, U1: T x d_2 x R, U3: T x d_3 x R
        pred = U[0][train_ind[:, -1], train_ind[:, 0]]
        for j in range(1, self.nmod):
            pred = pred * U[j][train_ind[:, -1], train_ind[:, j]]
        pred = torch.sum(pred, 1)
        ll = ll - 0.5 * torch.exp(-self.log_tau) * torch.sum(
            torch.square(train_y - pred)
        )
        return -ll

    def train(self, train_ind, train_y, test_ind, test_y):
        train_y = torch.tensor(train_y)
        test_y = torch.tensor(test_y)
        paras = [self.U, self.log_v, self.log_tau]
        optimizer = torch.optim.Adam(paras, lr=1e-3)
        for n in range(self.nepoch):
            loss = self.get_loss(train_ind, train_y)
            if n % 100 == 0:
                with torch.no_grad():
                    pred_test = self.pred(test_ind, test_y)
                    err = torch.sqrt(torch.mean((pred_test - test_y) ** 2))
                    print("epoch %d, Error u: %g" % (n, err))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        with torch.no_grad():
            pred_test = self.pred(test_ind, test_y)
            err = torch.sqrt(torch.mean((pred_test - test_y) ** 2))
            print("Error u: %g" % err)

    def pred(self, test_ind, test_y):
        # T x d x R
        U = torch.split(self.U, self.nvec.tolist(), dim=1)
        # U0: T x d_1 x R, U1: T x d_2 x R, U3: T x d_3 x R
        pred = U[0][test_ind[:, -1], test_ind[:, 0]]
        for j in range(1, self.nmod):
            pred = pred * U[j][test_ind[:, -1], test_ind[:, j]]
        pred = torch.sum(pred, 1)
        return pred


def test_beijing():
    ind = []
    y = []
    with open("./data/beijing_15k_train.txt", "r") as f:
        for line in f:
            items = line.strip().split(",")
            y.append(float(items[-1]))
            ind.append([int(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
    nvec = np.max(ind, 0) + 1

    test_ind = []
    test_y = []
    with open("./data/beijing_15k_test.txt", "r") as f:
        for line in f:
            items = line.strip().split(",")
            test_y.append(float(items[-1]))
            test_ind.append([int(idx) for idx in items[0:-1]])
        test_ind = np.array(test_ind)
        test_y = np.array(test_y)

    R = 2
    ns = nvec[-1]
    nvec = nvec[0:-1]
    nepoch = 20000
    model = CPD(nvec, R, ns, nepoch)
    model.train(ind, y, test_ind, test_y)


if __name__ == "__main__":
    test_beijing()
