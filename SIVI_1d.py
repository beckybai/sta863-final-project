# Coded by Mingzhang Yin
# 02/09/2018 latest version.

# Copyright (c) <2018> <Mingzhang Yin>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# %%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
import tensorflow as tf
import os

slim = tf.contrib.slim
Exponential = tf.contrib.distributions.Exponential(rate=1.0)
Normal1 = tf.contrib.distributions.Normal(loc=10.0, scale=1.0)
Normal2 = tf.contrib.distributions.Normal(loc=16.0, scale=1.0)
Normal = tf.contrib.distributions.Normal(loc=0., scale=1.)

directory = os.getcwd()


# %%

def sample_n(mu, sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu + eps * sigma
    return z


def sample_hyper(noise_dim, K, reuse=False):
    z_dim = 1
    with tf.variable_scope("hyper_q") as scope:
        if reuse:
            scope.reuse_variables()
        e2 = tf.random_normal(shape=[K, noise_dim])
        input_ = e2
        h2 = slim.stack(input_, slim.fully_connected, [20, 40, 20])
        mu = tf.reshape(slim.fully_connected(h2, z_dim, activation_fn=None, scope='implicit_hyper_mu'), [-1, 1])
    return mu


# %%

data_p = {"1": "gaussian", "2": "laplace", "3": "gmm"}
data_number = "3"
target = data_p[data_number]

# %%
noise_dim = 10
K = 30

psi_sample = sample_hyper(noise_dim, K)

sigma = tf.constant(0.2)
z_sample = sample_n(psi_sample, sigma)

J = tf.placeholder(tf.int32, shape=())
psi_star = tf.transpose(sample_hyper(noise_dim, J, reuse=True))

merge = tf.placeholder(tf.int32, shape=[])
psi_star = tf.cond(merge > 0, lambda: tf.concat([psi_star, tf.transpose(psi_sample)], 1), lambda: psi_star)

log_H = tf.log(tf.reduce_mean(tf.exp(-0.5 * tf.square(z_sample - psi_star) / tf.square(sigma)), axis=1, keep_dims=True))

# log_Q = -tf.log(sigma)-0.5*tf.square(z_sample-psi_sample)/tf.square(sigma)
# regular = log_Q - log_H

if target == 'gaussian':
    log_P = -tf.log(3.0) - 0.5 * tf.square(z_sample) / tf.square(3.0)  # gaussian
elif target == 'laplace':
    log_P = -0.5 * tf.abs(z_sample)  # laplace(mu=0,b=2)
elif target == 'gmm':
    log_P = tf.log(0.7 * tf.exp(-tf.square(z_sample - 10) / 2) + 0.3 * tf.exp(-tf.square(z_sample - 16) / 2))
else:
    raise ValueError('No pre-defined target distribution, you can write your own log(PDF) ')

loss = tf.reduce_mean(log_H - log_P)

nn_var = slim.get_model_variables()
lr = tf.constant(0.01)
train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=nn_var)

init_op = tf.global_variables_initializer()

# %%
# merge==1 corresponds to lower bound; merge==0 corresponds to upper bound
sess = tf.InteractiveSession()
sess.run(init_op)
record = []
for i in range(5000):
    _, cost = sess.run([train_op1, loss], {lr: 0.001, J: 100, merge: 1})
    record.append(cost)
    if i % 500 == 0:
        print("iter:", '%04d' % (i + 1), "cost=", np.mean(record))
        record = []
# %% plot Q and target
r_hive = []
for i in range(100):
    r = sess.run(z_sample)
    r_hive.extend(np.squeeze(r))

yy = []
xx = np.arange(0, 25, 0.01)
for r in xx:
    if target == 'gaussian':
        pdf = stats.norm.pdf(r, loc=0, scale=3)  # gaussian
    elif target == 'laplace':
        pdf = 1 / 4 * np.exp(-0.5 * np.abs(r))  # laplace
    elif target == 'gmm':
        pdf = 0.7 * stats.norm.pdf(r, loc=10, scale=1) + 0.3 * stats.norm.pdf(r, loc=16, scale=1)  # gmm
    yy.append(pdf)

ax = plt.figure()
ax = sns.distplot(r_hive, label='Q distribution')
ax = plt.plot(xx, yy, 'y-', label='P distribution')
plt.legend()

# %% plot latent
latent = []
zsamples = []
for i in range(200):
    muu = sess.run(psi_sample)
    latent.extend(np.squeeze(muu))

plt.figure()
import torch
import torch.distributions as dist
import seaborn as sns
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
plt.plot(x.numpy(),(y1*0.3+y2*0.7).numpy())
plt.plot(x.numpy(),y.numpy(),color='r')

sns.distplot(latent)
plt.title("Gaussian \mu_i")
