import numpy as np

from scipy.stats import multivariate_normal as mnormal
from scipy.io import loadmat
import torch
import os
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2019)
np.random.seed(2019)

torch.set_default_dtype(torch.float32)
def sigmoid(x):
    return 1/(1+torch.exp(-x))

def logit(x):
    return torch.log(x) - torch.log(1-x)

# matdata = loadmat(os.getcwd()+'/a9a.mat')
#
#
#
# X_train = matdata['X_train'][:,:-1]
# X_test = matdata['X_test']
# y_train = np.int_(np.squeeze(matdata['y_train']))
# y_test = np.int_(np.squeeze(matdata['y_test']))
#
#
# n_data,len_data = X_train.shape[0],X_train.shape[1]
#
#
#
# N, P = np.shape(X_train)
# Nt, P = np.shape(X_test)
#
# xtr = np.ones([N,P+1])
# xtr[:,:-1] = X_train
# xte = np.ones([Nt,P+1])
# xte[:,:-1] = X_test
#
#
#
# X_train = xtr
# X_test = xte
# N, len_data = np.shape(X_train)
#
# X_train = torch.Tensor(X_train)
# X_test = torch.Tensor(X_test)
# y_train =torch.Tensor(y_train)
# y_test = torch.Tensor(y_test)
#
# X_train = X_train.to(device)
# X_test = X_test.to(device)
# y_train = y_train.to(device)
# y_test = y_test.to(device)
#
# # randomize the data
# # TODO
d1 = dist.normal.Normal(torch.tensor([10.0]).to(device), torch.tensor([1.0]).to(device))
d2 = dist.normal.Normal(torch.tensor([16.0]).to(device), torch.tensor([1.0]).to(device))
len_data = 1

torch_zeros = torch.Tensor(torch.from_numpy(np.zeros([len_data,1]).astype('float32'))).to(device)
torch_eye = torch.Tensor(torch.from_numpy(np.eye(len_data).astype('float32'))).to(device)

def likelihood(beta,x):
    sigmoid(torch.mm(x, beta))
    return

def logpost(data):
    # llx = (likelihood(beta, X_train))
    # wx = sigmoid(torch.mm(data,beta))


    # return torch.sum(torch.log(llx+1e-6)*y_train) \
    #        + torch.sum(torch.log(1-llx+1e-6)*(1-y_train))\
    #        - torch.sum((data)**2/(2*torch.diag(torch_eye)))

    y1 = torch.exp(d1.log_prob(data))
    y2 = torch.exp(d2.log_prob(data))
    wx = torch.log(y1*0.3+y2*0.7)

    return wx
           # + dmn.log_prob(beta.reshape([-1]))


def proposal_dis():
    return 0.05*torch.randn(torch.Size([len_data,1])).to(device)
    # return torch.HalfTensor(np.random.randn(len_data,1).tolist()).to(device)

accept_list = []
total_iter = 50000

beta_list = []


def train(beta_current):
    accept_i = 0

    for i in range(total_iter):
        log_prob_curr = logpost(beta_current)
        propo = proposal_dis()
        log_prob_next = logpost(beta_current+propo)

        alpha = torch.clamp(torch.exp(log_prob_next-log_prob_curr),max=1)
        u = torch.rand(1).to(device)
        if(alpha>u):
            beta_current = beta_current + propo
            # beta_list[accept_i] = beta_current.squeeze()
            beta_list.append(beta_current.squeeze())
            accept_i +=1
            # accept_list.append(alpha)
        if(i%200==0) and len(beta_list)>0:
            print('alpha', alpha)




def predict(data,label, beta):
    # return np.mean(((sigmoid(X_test @ np.reshape(beta, [-1, 1]))).squeeze()-0.5) * (y_test-0.5) > 0)
    predict_l = (sigmoid(torch.mm(data, torch.reshape(beta, [-1, 1]))) - 0.5).squeeze().cpu().numpy()
    real_l = (label-0.5).cpu().numpy()
    print('tt', np.sum(np.logical_and(predict_l>0,real_l>0)))
    print('ft', np.sum(np.logical_and(predict_l<0,real_l>0)))
    print('tf', np.sum(np.logical_and(predict_l>0,real_l<0)))
    print('ff', np.sum(np.logical_and(predict_l<0,real_l<0)))


    return np.mean( (predict_l>0) == (real_l>0))
def remove_outlier(x, thresh=3.5):
    """
    Returns points that are not outliers to make histogram prettier
    reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
    Arguments:
        x {numpy.ndarray} -- 1d-array, points to be filtered
        thresh {float} -- the modified z-score to use as a threshold. Observations with
                          a modified z-score (based on the median absolute deviation) greater
                          than this value will be classified as outliers.
    Returns:
        x_filtered {numpy.ndarray} -- 1d-array, filtered points after dropping outlier
    """
    if len(x.shape) == 1: x = x[:,None]
    median = np.median(x, axis=0)
    diff = np.sqrt(((x - median)**2).sum(axis=-1))
    modified_z_score = 0.6745 * diff / np.median(diff)
    x_filtered = x[modified_z_score <= thresh]
    return x_filtered
beta_init =torch.Tensor(np.random.randn(len_data,1).tolist()).to(device)
# beta_init = np.array([-1.685791  ,  0.18304595,  0.30819154,  0.53551656,  0.32589892,
#        -0.06726686, -0.11718246,  0.7808014 ,  0.03881752, -0.8896701 ,
#        -0.07555294, -3.4123764 , -1.1800346 , -0.6103314 , -1.1518898 ,
#        -1.5954454 , -0.6154823 , -1.1323537 , -0.41858515,  1.2315726 ,
#        -1.0616033 , -1.1867207 , -0.49612796,  0.55465865, -2.27489   ,
#        -2.3791947 , -1.3860116 , -0.25156915,  0.85513556,  0.63100195,
#        -1.8627927 ,  3.5200517 , -0.98226595, -0.31264728, -1.340569  ,
#        -0.04547392, -1.7715547 ,  1.0410848 , -0.27713034,  1.5563657 ,
#        -1.0356903 , -0.7010865 , -2.7465262 , -1.9171436 , -1.1015661 ,
#         1.421215  , -0.4720165 , -0.27531558,  0.9815484 ,  0.3864525 ,
#         1.1069903 ,  1.1150465 , -1.3857496 , -0.5766556 ,  1.4755181 ,
#        -1.1800312 , -1.3491341 , -1.0718291 , -0.4117614 , -2.3519855 ,
#         1.3236177 , -0.613691  , -0.31389236,  0.7189452 , -0.86144596,
#         0.3199659 , -0.09741431,  0.1697346 , -1.4168288 , -0.5101275 ,
#        -0.016675  , -1.4301263 , -0.18919271, -1.0679373 ,  1.4445989 ,
#        -1.4637125 ,  1.6140019 , -1.4239218 , -0.4963808 ,  0.28591377,
#         1.1925335 ,  0.587555  ,  0.23757216,  1.9443498 ,  0.4632172 ,
#        -0.7187378 ,  0.7907927 ,  1.1204501 , -3.336615  ,  0.2719301 ,
#         0.16714638, -1.6792903 , -0.30321795, -1.897389  ,  1.1216239 ,
#        -0.850836  , -1.0430856 ,  1.0276406 , -0.02986836,  0.36239243,
#        -0.81673986,  0.17491937,  2.0132337 , -0.18327828,  2.4611511 ,
#         3.0154736 , -2.821262  , -1.6861777 , -0.66073704,  0.959491  ,
#        -1.3874034 , -4.0175695 , -0.2095837 , -0.8485173 , -1.9368992 ,
#        -0.9140258 , -2.2030072 ,  1.3023987 , -1.9462571 , -1.8205552 ,
#        -1.5483217 ,  0.90997964,  1.2894483 ], dtype='float32')
# beta_init = torch.Tensor(torch.from_numpy(beta_init.reshape([-1,1]))).to(device)

# beta_init = beta.reshape([-1,1])
train(beta_init)


# test0 = X_test[y_test==0].cpu().numpy()
# test1 = X_test[y_test==1].cpu().numpy()
# beta = beta_list[-1].cpu().numpy()
# a1 = remove_outlier(((test1 @ np.reshape(beta*2-1, [-1, 1]))).squeeze())
# a0 = remove_outlier(((test0 @ np.reshape(beta*2-1, [-1, 1]))).squeeze())
# output = [a0 ,a1]
# plt.figure(figsize=(8, 4))
# plt.hist(output, bins=50)
beta_list_np = np.zeros([len(beta_list)])
for i,bb in enumerate(beta_list):
    beta_list_np[i] = bb.cpu().numpy()


plt.figure()
d1 = dist.normal.Normal(torch.tensor([10.0]), torch.tensor([1.0]))
d2 = dist.normal.Normal(torch.tensor([16.0]), torch.tensor([1.0]))
w1 = 0.3
w2 = 0.7
d = dist.normal.Normal(torch.tensor([10.0]), torch.tensor([1.0]))
a = d.sample(torch.Tensor(torch.rand(100)).shape)
torch.mean(d.log_prob(a) - torch.log(torch.exp(d1.log_prob(a))*w1+torch.exp(d2.log_prob(a))*w2))

x = torch.Tensor(np.linspace(5,25,100))
y1 = torch.exp(d1.log_prob(x))
y2 = torch.exp(d2.log_prob(x))
y = torch.exp(d.log_prob(x))
plt.plot(x.numpy(),(y1*w1+y2*w2).numpy())

sns.distplot(beta_list_np)
plt.xlabel('x')
plt.ylabel('probability')
plt.legend(['target','approximate'])
plt.show()
# %% plot latent


plt.figure()
sns.distplot(beta_list_np)
plt.xlim([0,25])

print('can you hear me?')

print('233')