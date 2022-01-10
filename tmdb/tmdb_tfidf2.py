'''
Using random projections and by building up a tree. 
At every intermediate node in the tree, a random hyperplane is chosen, 
which divides the space into two subspaces. 
This hyperplane is chosen by sampling two points from the subset 
and taking the hyperplane equidistant from them

This way, similar entries will be closer on a tree

We do this k times so that we get a forest of trees
'''
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

# https://github.com/spotify/annoy
from annoy import AnnoyIndex

# get the data from: https://www.kaggle.com/tmdb/tmdb-movie-metadata
# load in the data
df = pd.read_csv('/tmp2/b07902053/tmdb_5000_movies.csv')
index_file = "index.ann"

# convert the relevant data for each movie into a single string
# to be ingested by TfidfVectorizer


def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return f"{genres} {keywords}"


# create a new string representation of each movie
df['string'] = df.apply(genres_and_keywords_to_string, axis=1)


print("extracting features")
# create a tf-idf vectorizer object
# remove stopwords automatically
# build a vocabulary that only consider the top max_features
# ordered by term frequency across the corpus
tfidf = TfidfVectorizer(max_features=2000)

# create a data matrix from the overviews
X = tfidf.fit_transform(df['string'])  # df['string'] is a ndarray

# check the shape of X
print("X.shape:", X.shape)  # (4803, 2000)


print("building index")
# "angular", "euclidean", "manhattan", "hamming", or "dot"
# "angular": sqrt(2(1-cos(u,v)))
# "dot": query-friendly cosine space
t = AnnoyIndex(X.shape[1], 'dot')
Xc = X.tolil()  # list in list format
for i, (r, d) in tqdm(enumerate(zip(Xc.rows, Xc.data))):
    feature = np.zeros(Xc.shape[1])
    for j, v in zip(r, d):
        feature[j] = v
    t.add_item(i, feature)

# A larger value (n_trees) will give more accurate results, but larger indexes
t.build(200)  # 200 trees
t.save(index_file)
print(f"saved to {index_file}")

# generate a mapping from movie title -> index (in df)
movie2idx = pd.Series(data=df.index, index=df['title'])


class FastApproxTFIDF(object):
    def __init__(self, dims, k, index_file):
        self.dims = dims
        self.k = k  # top k neighbors
        self.index_file = index_file
        self.t = AnnoyIndex(self.dims, 'dot')
        self.t.load(index_file)  # super fast, will just mmap the file

    def recommend(self, title):
        # get the row in the dataframe for this movie
        try:
            idx = movie2idx[title]
            # don't include itself
            # recommended_idx = self.t.get_nns_by_item(idx, self.k + 1)
            # return list(df['title'].iloc[recommended_idx][1:])

            # search_k is provided in runtime and affects the search performance.
            # A larger value will give more accurate results,
            # but will take longer time to return
            # default: n_trees * (# of desired neighbors) (search_k=-1)
            recommended_idx, scores = self.t.get_nns_by_item(idx, self.k + 1,
                                                             search_k=-1, include_distances=True)
            return [(title, score) for title, score in zip(df['title'].iloc[recommended_idx][1:], scores[1:])]

        except KeyError:
            return []

    def search_by_vector(self, v):
        recommended_idx, scores = self.t.get_nns_by_vector(v, self.k,
                                                           search_k=-1, include_distances=True)
        return [(title, score) for title, score in zip(df['title'].iloc[recommended_idx], scores)]


if __name__ == '__main__':
    rec = FastApproxTFIDF(X.shape[1], 5, index_file)

    print("\nTest for not existed input")
    print(rec.recommend('BlaBlaBlaBlaBlaBlaBlaBla'))

    print("\nRecommendations for 'Scream 3':")
    print(rec.recommend('Scream 3'))

    print("\nRecommendations for 'Mortal Kombat':")
    print(rec.recommend('Mortal Kombat'))

    print("\nRecommendations for 'Runaway Bride':")
    print(rec.recommend('Runaway Bride'))

    print("\nRecommendations for 'The Dark Knight Rises':")
    print(rec.recommend('The Dark Knight Rises'))
