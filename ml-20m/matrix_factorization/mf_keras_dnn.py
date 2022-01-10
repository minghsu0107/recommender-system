import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
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
# reg = 0.0001 # regularization penalty


# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u)  # (None, 1, K)
m_embedding = Embedding(M, K)(m)  # (None, 1, K)
u_embedding = Flatten()(u_embedding)  # (None, K)
m_embedding = Flatten()(m_embedding)  # (None, K)
x = Concatenate()([u_embedding, m_embedding])  # (None, 2K)

# the neural network
x = Dense(400)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(100)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(1)(x)

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
fig.savefig('./mf-keras-deep-loss.png')

# plot mse
fig, ax = plt.subplots(1)
ax.plot(epo, r.history['mse'], label="train mse")
ax.plot(epo, r.history['val_mse'], label="test mse")
ax.legend()
fig.savefig('./mf-keras-deep-error.png')
plt.close()
