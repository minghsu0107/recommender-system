# note that A and A_test are both (N, M), but A_test will be sparser (contains more zeros)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

# load in the data
df = pd.read_csv('/tmp2/b07902053/ml-20m/edited_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

A = lil_matrix((N, M))
print("Calling: update_train")
count = 0


def update_train(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / cutoff))

    i = int(row.userId)
    j = int(row.movie_idx)
    A[i, j] = row.rating


df_train.apply(update_train, axis=1)

# lil: better for adding values
# csr: better for saving
A = A.tocsr()
# mask, to tell us which entries exist and which do not
# mask = (A > 0)
save_npz("/tmp2/b07902053/ml-20m/Atrain.npz", A)

# test ratings dictionary
A_test = lil_matrix((N, M))
print("Calling: update_test")
count = 0


def update_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / len(df_test)))

    i = int(row.userId)
    j = int(row.movie_idx)
    A_test[i, j] = row.rating


df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()
# mask_test = (A_test > 0)
save_npz("/tmp2/b07902053/ml-20m/Atest.npz", A_test)
