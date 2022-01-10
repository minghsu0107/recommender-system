import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('/tmp2/b07902053/ml-20m/small_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10  # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0.  # regularization penalty


# keras model
u = Input(shape=(1,))  # Length of input sequences, here it's 1 (rating)
m = Input(shape=(1,))
# Embedding(input_dim, output_dim)
# input_dim: Size of the vocabulary
# output_dim: Dimension of the dense embedding
u_embedding = Embedding(
    N, K, embeddings_regularizer=l2(reg))(u)  # (None, 1, K)
m_embedding = Embedding(
    M, K, embeddings_regularizer=l2(reg))(m)  # (None, 1, K)

# subsubmodel = Model([u, m], [u_embedding, m_embedding])
# user_ids = df_train.userId.values[0:5]
# movie_ids = df_train.movie_idx.values[0:5]
# print("user_ids.shape", user_ids.shape) # (5,)
# p = subsubmodel.predict([user_ids, movie_ids])
# print("p[0].shape:", p[0].shape) # (5, 1, 10)
# print("p[1].shape:", p[1].shape) # (5, 1, 10)
# exit()


u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u)  # (None, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m)  # (None, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding])  # (None, 1, 1)

# submodel = Model([u, m], x)
# user_ids = df_train.userId.values[0:5]
# movie_ids = df_train.movie_idx.values[0:5]
# p = submodel.predict([user_ids, movie_ids])
# print("p.shape:", p.shape) # (5, 1, 1)
# exit()


x = Add()([x, u_bias, m_bias])
x = Flatten()(x)  # (None, 1*1)

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss='mse',
    # optimizer='adam',
    # optimizer=Adam(lr=0.01),
    optimizer=SGD(lr=0.08, momentum=0.9),
    metrics=['mse'],
)

r = model.fit(
    x=[df_train.userId.values, df_train.movie_idx.values],
    y=df_train.rating.values - mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test.userId.values, df_test.movie_idx.values],
        df_test.rating.values - mu
    )
)

# plot losses
epo = range(1, epochs + 1)
fig, ax = plt.subplots(1)
ax.plot(epo, r.history['loss'], label="train loss")
ax.plot(epo, r.history['val_loss'], label="test loss")
ax.legend()
fig.savefig('./mf-keras-loss.png')

# plot mse
fig, ax = plt.subplots(1)
ax.plot(epo, r.history['mse'], label="train mse")
ax.plot(epo, r.history['val_mse'], label="test mse")
ax.legend()
fig.savefig('./mf-keras-error.png')
plt.close()
