import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('/tmp2/b07902053/ml-20m/ratings.csv')  # (20000263, 4)

'''
userId movieId rating timestamp
1      2       3.5    1112486027
1      29      3.5    1112484676
1      32      3.5    1112484819
1      47      3.5    1112484727
.
.
.
'''
# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('/tmp2/b07902053/ml-20m/edited_rating.csv', index=False)
