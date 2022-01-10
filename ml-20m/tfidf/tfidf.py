import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

df = pd.read_csv('/tmp2/b07902053/ml-20m/itdf.csv')
df1 = pd.read_csv('/tmp2/b07902053/ml-20m/itdf1.csv')

indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()

print("vectorizing...")

# analyzer="word": break words by spaces
# default token_pattern is '(?u)\b\w\w+\b'
# The default regexp selects tokens of 2 or more alphanumeric characters,
# and punctuation is completely ignored and always treated as a token separator ("|" in genres, etc.)

# ngram can be 2: for handling genres like "comedy romance", "documentary imax"
# min_df=1: ignore terms that have a document frequency strictly lower than 1
tfv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
X = tfv.fit_transform(df['genres'])
model = sigmoid_kernel(X, X)


def recommendations(title):
    idx = indices[title]
    dis_scores = list(enumerate(model[idx]))
    dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)
    dis_scores = dis_scores[1:8]

    idn = [i[0] for i in dis_scores]
    print(list(df['title'].iloc[idn]))


if __name__ == '__main__':
    recommendations('Toy Story')
    print("")
    recommendations('Before and After')
