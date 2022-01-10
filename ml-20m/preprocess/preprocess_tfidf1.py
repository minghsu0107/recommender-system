import numpy as np
import pandas as pd

print("reading data...")
movie = pd.read_csv('/tmp2/b07902053/ml-20m/movies.csv')
rating = pd.read_csv('/tmp2/b07902053/ml-20m/ratings.csv')
print(movie.shape)  # (27278, 3)
print(rating.shape)  # (20000263, 4)
print(movie.iloc[0]['title'])  # Toy Story (1995)

print("processing data...")
# inner join, since there may be multiple users rating on a single movie
# hence the df may contain entries with same movieId but different user
movie_details = movie.merge(rating, on='movieId')
movie_details.drop(columns=['timestamp'], inplace=True)
print(movie_details.shape)  # (20000263, 5)
'''
   movieId             title  ... userId  rating
0        1  Toy Story (1995)  ...      3     4.0
1        1  Toy Story (1995)  ...      6     5.0
2        1  Toy Story (1995)  ...      8     4.0
3        1  Toy Story (1995)  ...     10     4.0
4        1  Toy Story (1995)  ...     11     4.5
'''

# movie_details.groupby(['movieId', 'genres']).sum() is a df:
'''
                                                         userId    rating
movieId genres
1       Adventure|Animation|Children|Comedy|Fantasy  3442988710  194866.0
2       Adventure|Children|Fantasy                   1538546713   71444.0
3       Comedy|Romance                                879632931   40128.5
4       Comedy|Drama|Romance                          191963429    7886.0
5       Comedy                                        840488975   37268.5
'''
# consider duplicated ratings on same movies
# drop the orginal index
total_ratings = movie_details.groupby(['movieId', 'genres']).sum()[
    'rating'].reset_index(drop=True)
print(total_ratings.shape)  # (26744, 3)
'''
       movieId                                       genres    rating
0            1  Adventure|Animation|Children|Comedy|Fantasy  194866.0
1            2                   Adventure|Children|Fantasy   71444.0
2            3                               Comedy|Romance   40128.5
3            4                         Comedy|Drama|Romance    7886.0
4            5                                       Comedy   37268.5
'''

df = movie_details.copy()
# this shows that there are different movieId entries but with same title and genres
# (26739, 5), index is automatically reset
df.drop_duplicates(['title', 'genres'], inplace=True)
df = df.merge(total_ratings, on='movieId')  # (26739, 7)
print(df.columns)
# Index(['movieId', 'title', 'genres_x', 'userId', 'rating_x', 'genres_y', 'rating_y'], dtype='object')


df.drop(columns=['userId', 'rating_x', 'genres_y'], inplace=True)
df.rename(columns={'genres_x': 'genres', 'rating_y': 'rating'}, inplace=True)
df['rating'] = df['rating'].astype(int)
df = df[df['rating'] > 25]
print(df.shape)  # (15951, 4)

df.to_csv('/tmp2/b07902053/ml-20m/itdf.csv', index=False)  # no index
print("write to /tmp2/b07902053/ml-20m/itdf.csv")
