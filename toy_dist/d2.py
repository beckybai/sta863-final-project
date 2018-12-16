import numpy as np
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats


class toy_dist():
    def __init__(self, data_number='8'):
        data_p = {"5": "normal2d", "6": "gmm2d", "7": "banana", "8": "Xshape","9":'banana2',"10":"banana_shift"}
        self.target = data_p[data_number]
    
    def log_prob(self, z_sample):
        if self.target == "normal2d":
            log_P = -0.5 * tf.reduce_sum(tf.square(z_sample), 1, keep_dims=True) / (2 ** 2)
        elif self.target == "gmm2d":
            log_P = tf.log(0.5 * tf.exp(-0.5 * tf.reduce_sum(tf.square(z_sample - tf.constant([-2., 0.])), 1, keep_dims=True)) + \
                           0.5 * tf.exp(-0.5 * tf.reduce_sum(tf.square(z_sample - tf.constant([2., 0.])), 1, keep_dims=True)))
        elif self.target == "banana":
            z1 = tf.slice(z_sample, [0, 0], [-1, 1])
            z2 = tf.slice(z_sample, [0, 1], [-1, 1])
            sg1 = 2.0
            sg2 = 1.0
            log_P = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square(z2 - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)

        elif self.target == "banana2":
            z1 = tf.slice(z_sample, [0,0],[-1,1])
            z2 = tf.slice(z_sample, [0,1],[-1,1])
            sg1 = 2.0
            sg2 = 1.0
            log_P1 = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square((z2-10) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)
            log_P2 = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square(-(z2+10) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)
            X =tf.concat([log_P1, log_P2], -1)
            log_P = tf.reduce_logsumexp(X,1)-tf.log(2.0)


        elif self.target == "banana_shift":
            z1 = tf.slice(z_sample, [0,0],[-1,1])
            z2 = tf.slice(z_sample, [0,1],[-1,1])
            sg1 = 2.0
            sg2 = 1.0
            log_P1 = tf.clip_by_value(-0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square((z2-5) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2),-50,2)
            log_P2 =tf.clip_by_value(-0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square(-(z2+5) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2),-50,2)
            log_P = tf.log(0.5*tf.exp(log_P1) + 0.5*tf.exp(log_P2))


        elif self.target == "banana4":
            z1 = tf.slice(z_sample, [0,0],[-1,1])
            z2 = tf.slice(z_sample, [0,1],[-1,1])
            sg1 = 2.0
            sg2 = 1.0
            log_P1 = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square((z2-5) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)
            log_P2 = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square(-(z2+5) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)

            log_P3 = -0.5 * tf.square(z2) / tf.square(sg1) - 0.5 * tf.square((z1-5) - 0.25 * tf.square(z2)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)
            log_P4 = -0.5 * tf.square(z1) / tf.square(sg1) - 0.5 * tf.square(-(z2+5) - 0.25 * tf.square(z1)) / tf.square(sg2) - \
                    tf.log(sg1) - tf.log(sg2)

            X = tf.tile(tf.expand_dims(log_P1,0),[2,tf.shape(log_P1)[0],tf.shape(log_P1)[1]])

            log_P = tf.reduce_logsumexp(X,axis=0)


        elif self.target == "Xshape":
            def bi_gs(z1, z2, v, c):
                a = tf.square(v) - tf.square(c)
                b = -0.5 * (v * tf.square(z1) + v * tf.square(z2) - 2 * c * z1 * z2) / a
                return -0.5 * tf.log(a) + b


            z1 = tf.slice(z_sample, [0, 0], [-1, 1])
            z2 = tf.slice(z_sample, [0, 1], [-1, 1])
            log_P = tf.log(0.5 * tf.exp(bi_gs(z1, z2, 2.0, 1.8)) + 0.5 * tf.exp(bi_gs(z1, z2, 2.0, -1.8)))
        else:
            raise ValueError('No pre-defined self.target distribution, you can write your own log(PDF) ')
        if(self.target=='banana_shift'):
            return log_P1, log_P2, log_P

        return log_P

    # plot the groundtruth


    def plot_contous(self):
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X;
        pos[:, :, 1] = Y

        if self.target == 'normal2d':
            rv = stats.multivariate_normal(mean=[0, 0], cov=[[4, 0], [0, 4]])
            Z = rv.pdf(pos)
            CS = plt.contour(X, Y, rv.pdf(pos), 5, colors='b')

        elif self.target == 'gmm2d':
            rv1 = stats.multivariate_normal(mean=[-2, 0], cov=[[1, 0], [0, 1]])
            rv2 = stats.multivariate_normal(mean=[2, 0], cov=[[1, 0], [0, 1]])
            Z = 0.5 * rv1.pdf(pos) + 0.5 * rv2.pdf(pos)
            CS = plt.contour(X, Y, Z, 5, colors='b')

        elif self.target == 'sin':
            Z = np.exp(-0.5 * np.square((Y - np.sin(X)) / 0.4))
            CS = plt.contour(X, Y, Z, 5, colors='b')

        elif self.target == 'banana':
            X, Y = np.mgrid[-5:5:.01, -5:5:.01]
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X;
            pos[:, :, 1] = Y

            sg1 = 2;
            sg2 = 1
            Z = (1 / sg1) * np.exp(-0.5 * X ** 2 / (sg1 ** 2)) * (1 / sg2) * np.exp(
                -0.5 * ((Y - 0.25 * X * X) ** 2) / (sg2 ** 2))
            CS = plt.contour(X, Y, Z, 5, colors='b')

        elif self.target == 'banana2':
            X, Y = np.mgrid[-5:5:.01, -20:20:.01]
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X;
            pos[:, :, 1] = Y

            sg1 = 2;
            sg2 = 1
            Z1 = (1 / sg1) * np.exp(-0.5 * X ** 2 / (sg1 ** 2)) * (1 / sg2) * np.exp(
                -0.5 * (((Y - 10) - 0.25 * X * X) ** 2) / (sg2 ** 2))

            Z2 = (1 / sg1) * np.exp(-0.5 * X ** 2 / (sg1 ** 2)) * (1 / sg2) * np.exp(
                -0.5 * ((-(Y + 10) - 0.25 * X * X) ** 2) / (sg2 ** 2))

            Z = 0.5 * Z1 + 0.5 * Z2

            CS = plt.contour(X, Y, Z, 10, cmap='winter')

        elif self.target =='banana_shift':
            X, Y = np.mgrid[-10:10:.01, -20:20:.01]
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X;
            pos[:, :, 1] = Y

            sg1 = 2;
            sg2 = 1
            Z1 = (1 / sg1) * np.exp(-0.5 * X ** 2 / (sg1 ** 2)) * (1 / sg2) * np.exp(
                -0.5 * (((Y - 5) - 0.25 * X * X) ** 2) / (sg2 ** 2))

            Z2 = (1 / sg1) * np.exp(-0.5 * X ** 2 / (sg1 ** 2)) * (1 / sg2) * np.exp(
                -0.5 * ((-(Y + 5) - 0.25 * X * X) ** 2) / (sg2 ** 2))

            Z = 0.5 * Z1 + 0.5 * Z2
            CS = plt.contour(X, Y, Z, 10, cmap='winter')

        elif self.target == 'Xshape':
            rv1 = stats.multivariate_normal(mean=[0, 0], cov=[[2, 1.8], [1.8, 2]])
            rv2 = stats.multivariate_normal(mean=[0, 0], cov=[[2, -1.8], [-1.8, 2]])
            Z = 0.5 * rv1.pdf(pos) + 0.5 * rv2.pdf(pos)
            CS = plt.contour(X, Y, Z, 5, colors='b')
