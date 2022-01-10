import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import SGD

# config
batch_size = 128
epochs = 20
reg = 0.0001
# reg = 0

A = load_npz("/tmp2/b07902053/ml-20m/Atrain.npz")
A_test = load_npz("/tmp2/b07902053/ml-20m/Atest.npz")
mask = (A > 0) * 1.0
mask_test = (A_test > 0) * 1.0

# make copies since we will shuffle
A_copy = A.copy()
mask_copy = mask.copy()
A_test_copy = A_test.copy()
mask_test_copy = mask_test.copy()

N, M = A.shape
print("N:", N, "M:", M)
print("N // batch_size:", N // batch_size)

# center the data
mu = A.sum() / mask.sum()
print("mu:", mu)


# build the model - just a 1 hidden layer autoencoder
i = Input(shape=(M,))
# bigger hidden layer size seems to help!
x = Dropout(0.7)(i)  # teach the model not to reconstruct but predict ratings
x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)
# x = Dropout(0.5)(x)
x = Dense(M, kernel_regularizer=l2(reg))(x)


def custom_loss(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    diff = y_pred - y_true
    sqdiff = diff * diff * mask  # sqdiff, mask: (None, M)
    sse = K.sum(K.sum(sqdiff))  # K.sum(sqdiff), K.sum(mask): (None,)
    n = K.sum(K.sum(mask))
    return sse / n


def generator(A, M):
    while True:
        A, M = shuffle(A, M)
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            a = a - mu * m  # must keep zeros at zero!
            # m2 = (np.random.random(a.shape) > 0.5)
            # noisy = a * m2
            noisy = a  # no noise here, since we haved done dropout in the model
            yield noisy, a


def test_generator(A, M, A_test, M_test):
    # A and A_test are in corresponding order (i = 0 ... N - 1f, j = 0 ... M - 1)
    # use train data as input for prediction,
    # compare predicted values with corresponding test data and mask others
    # both of size N x M
    while True:
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            at = A_test[i * batch_size:upper].toarray()
            mt = M_test[i * batch_size:upper].toarray()
            a = a - mu * m
            at = at - mu * mt
            yield a, at


model = Model(i, x)
model.compile(
    loss=custom_loss,
    optimizer=SGD(lr=0.08, momentum=0.9),
    # optimizer='adam',
    metrics=[custom_loss],
)

r = model.fit_generator(
    generator(A, mask),
    validation_data=test_generator(
        A_copy, mask_copy, A_test_copy, mask_test_copy),
    epochs=epochs,
    steps_per_epoch=A.shape[0] // batch_size + 1,
    validation_steps=A_test.shape[0] // batch_size + 1,
)
print(r.history.keys())


# plot losses
epo = range(1, epochs + 1)
fig, ax = plt.subplots(1)
ax.plot(epo, r.history['loss'], label="train loss")
ax.plot(epo, r.history['val_loss'], label="test loss")
ax.legend()
fig.savefig('./autorec-loss.png')

# plot mse
fig, ax = plt.subplots(1)
ax.plot(epo, r.history['custom_loss'], label="train custom_loss")
ax.plot(epo, r.history['val_custom_loss'], label="test custom_loss")
ax.legend()
fig.savefig('./autorec-error.png')
plt.close()
