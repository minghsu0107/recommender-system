# optimization:
# do one-hot encoding in tf
# do SSE calculation in tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


##### Helper Functions #####
def dot1(V, W):
    # V is N x D x K (batch of visible units)
    # W is D x K x M (weights)
    # returns N x M (hidden layer size)
    return tf.tensordot(V, W, axes=[[1, 2], [0, 1]])


def dot2(H, W):
    # H is N x M (batch of hiddens)
    # W is D x K x M (weights transposed)
    # returns N x D x K (visible)
    return tf.tensordot(H, W, axes=[[1], [2]])


class RBM(object):
    def __init__(self, D, M, K):
        self.D = D  # input feature size
        self.M = M  # hidden size
        self.K = K  # number of ratings
        self.build(D, M, K)

    def build(self, D, M, K):
        ##### Variables #####
        # trainable params
        self.W = tf.Variable(tf.random.normal(
            shape=(D, K, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))

        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))

        # one hot encode X
        # first, make each rating an int
        X = tf.cast(self.X_in * 2 - 1, tf.int32)
        X = tf.one_hot(X, K)

        ##### Hidden Layer Calculation #####
        # conditional probabilities
        # NOTE: tf.contrib.distributions.Bernoulli API has changed in Tensorflow v1.2
        V = X
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)  # (N, M)
        self.p_h_given_v = p_h_given_v  # save for later

        ##### Visible Layer Calculation #####
        # draw a sample from p(h | v)
        r = tf.random.uniform(shape=tf.shape(p_h_given_v))
        # (N, M), actual values (0 or 1), according to the prob
        H = tf.to_float(r < p_h_given_v)

        # draw a sample from p(v | h) (approximate p(v) by p(v | h))
        # note: we don't have to actually do the softmax to get the probabilities for sampling
        # we can use tf.distributions.Categorical for sampling
        logits = dot2(H, self.W) + self.b  # logits are raw scores
        cdist = tf.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()  # shape is (N, D): the actual sampled categories for D movies
        # transform to (N, D, K) to calculatie free energy
        X_sample = tf.one_hot(X_sample, depth=K)

        # mask X_sample to remove missing ratings
        mask2d = tf.cast(self.X_in > 0, tf.float32)
        # repeat K times in last dimension
        mask3d = tf.stack([mask2d]*K, axis=-1)
        X_sample = X_sample * mask3d

        ##### Objective and Minimization #####
        # build the objective (calculate the mean loss of the current batch)
        objective = tf.reduce_mean(self.free_energy(
            X)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)

        # build the cost (cross entropy)
        # we won't use this to optimize the model parameters
        # just to observe what happens during training
        logits = self.forward_logits(X)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=X,
                logits=logits,
            )
        )
        # to get the final output
        self.output_visible = self.forward_output(X)  # (N, D, K)

        ##### Calculate SSE (sum of square error) #####
        # train SSE
        self.one_to_ten = tf.constant(
            (np.arange(10) + 1).astype(np.float32) / 2)  # (10,)
        self.pred = tf.tensordot(
            self.output_visible, self.one_to_ten, axes=[[2], [0]])
        mask = tf.cast(self.X_in > 0, tf.float32)
        se = mask * (self.X_in - self.pred) * (self.X_in - self.pred)
        self.sse = tf.reduce_sum(se)

        # test SSE
        self.X_test = tf.placeholder(tf.float32, shape=(None, D))
        mask = tf.cast(self.X_test > 0, tf.float32)
        tse = mask * (self.X_test - self.pred) * (self.X_test - self.pred)
        self.tsse = tf.reduce_sum(tse)

        ##### Initialization #####
        initop = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initop)

    ##### Training #####

    def fit(self, X, X_test, epochs=10, batch_sz=256, show_fig=True):
        N, D = X.shape
        n_batches = N // batch_sz

        costs = []
        test_costs = []
        for i in range(epochs):
            # training process
            t0 = datetime.now()
            print("epoch:", i)
            # everything has to be shuffled accordingly
            X, X_test = shuffle(X, X_test)
            for j in range(n_batches):
                x = X[j * batch_sz: (j * batch_sz + batch_sz)].toarray()

                _, c = self.session.run(
                    (self.train_op, self.cost),
                    feed_dict={self.X_in: x}
                )

                if j % 100 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
            print("duration:", datetime.now() - t0)

            # calculate the true train and test cost
            t0 = datetime.now()
            sse = 0
            test_sse = 0
            n = 0
            test_n = 0
            for j in range(n_batches):
                # the test PREDICTIONS come from the train data!
                # X_test and mask_test are only used for targets
                x = X[j * batch_sz: (j * batch_sz + batch_sz)].toarray()
                xt = X_test[j * batch_sz: (j * batch_sz + batch_sz)].toarray()

                # number of train ratings
                n += np.count_nonzero(x)

                # number of test ratings
                test_n += np.count_nonzero(xt)

                # use tensorflow to get SSEs
                sse_j, tsse_j = self.get_sse(x, xt)
                sse += sse_j
                test_sse += tsse_j
            c = sse / n
            ct = test_sse / test_n
            print("train mse:", c)
            print("test mse:", ct)
            print("calculate cost duration:", datetime.now() - t0)
            costs.append(c)
            test_costs.append(ct)

        if show_fig:
            plt.plot(costs, label='train mse')
            plt.plot(test_costs, label='test mse')
            plt.legend()
            plt.show()

    ##### Private Methods #####

    def free_energy(self, V):
        first_term = -tf.reduce_sum(dot1(V, self.b))
        second_term = -tf.reduce_sum(
            # compute softplus: log(exp(features) + 1)
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            tf.nn.softplus(dot1(V, self.W) + self.c),
            axis=1
        )
        return first_term + second_term

    def forward_hidden(self, X):
        return tf.nn.sigmoid(dot1(X, self.W) + self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return dot2(Z, self.W) + self.b

    # return (N, D, K) probability distributions
    def forward_output(self, X):
        return tf.nn.softmax(self.forward_logits(X))

    ##### Public Methods #####
    # obtain a (N, M) feature vector (values in [0, 1])

    def transform(self, X):
        # accepts and returns a real numpy array
        # unlike forward_hidden and forward_output
        # which deal with tensorflow variables
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

    # get final prediction (N, D, K) (probabilities)
    def get_visible(self, X):
        return self.session.run(self.output_visible, feed_dict={self.X_in: X})

    # get final prediction (N, D, K) (actual ratings)
    def predict(self, X):
        return self.session.run(self.pred, feed_dict={self.X_in: X})

    # given X_train and X_test, return train and test error
    def get_sse(self, X, Xt):
        return self.session.run(
            (self.sse, self.tsse),
            feed_dict={
                self.X_in: X,
                self.X_test: Xt,
            })


def main():
    A = load_npz("/tmp2/b07902053/ml-20m/Atrain.npz")
    A_test = load_npz("/tmp2/b07902053/ml-20m/Atest.npz")

    N, M = A.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A, A_test)


if __name__ == '__main__':
    main()
