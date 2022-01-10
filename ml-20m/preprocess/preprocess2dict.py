import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# load in the data
# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('/tmp2/b07902053/ml-20m/small_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


# a dictionary to tell us which users have rated which movies
user2movie = {}
# a dicationary to tell us which movies have been rated by which users
movie2user = {}
# a dictionary to look up ratings
usermovie2rating = {}


print("Calling: update_user2movie_and_movie2user")
count = 0


def update_user2movie_and_movie2user(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))

    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating


df_train.apply(update_user2movie_and_movie2user, axis=1)

# test ratings dictionary
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0


def update_usermovie2rating_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/len(df_test)))

    i = int(row.userId)
    j = int(row.movie_idx)
    usermovie2rating_test[(i, j)] = row.rating


df_test.apply(update_usermovie2rating_test, axis=1)

# note: these are not really JSONs
with open('/tmp2/b07902053/ml-20m/user2movie.pkl', 'wb') as f:
    pickle.dump(user2movie, f)

with open('/tmp2/b07902053/ml-20m/movie2user.pkl', 'wb') as f:
    pickle.dump(movie2user, f)

with open('/tmp2/b07902053/ml-20m/usermovie2rating.pkl', 'wb') as f:
    pickle.dump(usermovie2rating, f)

with open('/tmp2/b07902053/ml-20m/usermovie2rating_test.pkl', 'wb') as f:
    pickle.dump(usermovie2rating_test, f)
