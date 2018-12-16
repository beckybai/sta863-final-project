import numpy as np
from scipy.stats import multivariate_normal as mnormal
from scipy.io import loadmat
import os
from toy_dist.d2 import toy_dist
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x) - np.log(1-x)



# def logpost(beta):
#     return np.sum(np.log(likelihood(beta, data_train_x))*data_train_y \
#            + np.log(1-likelihood(beta, data_train_x))*(1-data_train_y))\
#            + mnormal.logpdf(beta.reshape([-1]), np.zeros([len_data]),np.eye(len_data))

def plot_contous(target = 'banana2'):
    plot_xl = 0
    plot_yl = 20
    X, Y = np.mgrid[0:10:.1, -plot_yl:plot_yl:.1]
    pos = np.empty([X.shape[0]*X.shape[1],2])
    pos[:, 0] = X.reshape([-1])
    pos[:, 1] = Y.reshape([-1])

    if target == 'banana2':
        Z = logpost(pos).reshape(X.shape[0], Y.shape[1])
        CS = plt.contour(X, Y, Z, 5, colors='b')

def logpost(data):
    sg1 = 2.0
    sg2 = 1.0
    if(len(data)==2):
        z1,z2 = data[0],data[1]
    else:
        z1,z2 = data[:,0].reshape([-1,1]),data[:,1].reshape([-1,1])

    z1 = z1-5
    # z2 = z2-5
    log_P1 = -0.5 * z1**2 / sg1**2 - 0.5 * ((z2 - 10) - 0.25 * z1**2)**2 / (
        sg2) -  np.log(sg1) - np.log(sg2)**2
    log_P2 = -0.5 * z1**2 / sg1**2 - 0.5 * (-(z2 + 10) - 0.25 * z1**2)**2 / (
        sg2) -  np.log(sg1) - np.log(sg2)**2

    X = np.concatenate([log_P1, log_P2], -1)

    if(len(data)==2):
        log_P = np.log(np.mean(np.exp(X)))
    else:
        log_P = (np.mean(np.exp(X),axis=1))
    return log_P


def proposal_dis():
    return np.random.randn(2,1)*0.1

beta_list = []
accept_list = []
total_iter = 50000


def train(beta_current):
    for i in range(total_iter):
        log_prob_curr = logpost(beta_current)
        propo = proposal_dis()
        log_prob_next = logpost(beta_current+propo)

        alpha = np.min([1,np.exp(log_prob_next-log_prob_curr)])
        u = np.random.rand(1)
        if(alpha>u):
            beta_current = beta_current + propo
            beta_list.append(beta_current)
        accept_list.append(alpha)



beta_init = np.random.randn(2,1)
# beta_init = beta.reshape([-1,1])
train(beta_init)

beta_list_np = np.zeros([len(beta_list),2])
for i,vec in enumerate(beta_list):
    beta_list_np[i] = vec.squeeze()

plt.scatter(beta_list_np[:,0], beta_list_np[:,1],alpha = 0.01, color='r')
plot_contous(target = 'banana2')
plt.show()
print('233')