import pickle
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from tqdm import tqdm

user2movie = {}
movie2user = {}
usermovie2rating = {}
usermovie2rating_test = {}

N = 0
M = 0

K = 25
limit = 5

common_movies = {}
averages = []  # each user's average rating for later use
deviations = []  # each user's deviation for later use
sigma = []
weights = {}
neighbors = []  # store neighbors in this list


def load():
    print("loading preprocessed data...")
    global user2movie
    global movie2user
    global usermovie2rating
    global usermovie2rating_test
    global N
    global M

    with open('/tmp2/b07902053/ml-20m/user2movie.pkl', 'rb') as f:
        user2movie = pickle.load(f)

    with open('/tmp2/b07902053/ml-20m/movie2user.pkl', 'rb') as f:
        movie2user = pickle.load(f)

    with open('/tmp2/b07902053/ml-20m/usermovie2rating.pkl', 'rb') as f:
        usermovie2rating = pickle.load(f)

    with open('/tmp2/b07902053/ml-20m/usermovie2rating_test.pkl', 'rb') as f:
        usermovie2rating_test = pickle.load(f)

    N = np.max(list(user2movie.keys())) + 1

    if N > 10000:
        print("N =", N, "are you sure you want to continue?")
        print("Comment out these lines if so...")
        exit()
    # the test set may contain movies the train set doesn't have data on
    m1 = np.max(list(movie2user.keys()))
    m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
    M = max(m1, m2) + 1

    print("N:", N, "M:", M)


def precompute():
    print("precomputing...")
    movies_set = []
    ratings = []
    for i in tqdm(range(N)):
        movies_set.append(set(user2movie[i]))
        ratings.append(
            {movie: usermovie2rating[(i, movie)] for movie in user2movie[i]})
        averages.append(np.mean(list(ratings[i].values())))
        deviations.append(
            {movie: (rating - averages[i]) for movie, rating in ratings[i].items()})

        dev_i_values = np.array(list(deviations[i].values()))
        sigma.append(np.sqrt(dev_i_values.dot(dev_i_values)))

    print("calculating common_movies...")
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            common_movies[(i, j)] = (movies_set[i] & movies_set[j])
            common_movies[(j, i)] = common_movies[(i, j)]


# K: number of neighbors we'd like to consider
# limit: number of common movies users must have in common in order to consider
def getWeights(K, limit):
    print("computing weights...")
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            if j == i:
                continue
            if len(common_movies[(i, j)]) <= limit:
                continue

            num = sum(deviations[i][m] * deviations[j][m]
                      for m in common_movies[(i, j)])
            weights[(i, j)] = num / (sigma[i] * sigma[j])
            weights[(j, i)] = weights[(i, j)]


def getNeighbors(K):
    print("obtaining nearest neighbors...")
    for i in tqdm(range(N)):
        sl = SortedList()
        for j in range(N):
            if j == i:
                continue
            if (i, j) not in weights:
                continue
            # insert into sorted list and truncate
            # negate weight, because list is sorted ascending
            # maximum value (1) is "closest"
            sl.add((-weights[(i, j)], j))
            if len(sl) > K:
                del sl[-1]
        # store the neighbors
        neighbors.append(sl)


def predict(i, m):
    global neighbors
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        # remember, the weight is stored as its negative
        # so the negative of the negative weight is the positive weight
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # neighbor may not have rated the same movie
            # don't want to do dictionary lookup twice
            # so just throw exception
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)  # min rating is 0.5
    return prediction

# calculate accuracy


def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)


def main():
    load()
    precompute()
    getWeights(K, limit)
    getNeighbors(K)

    print("obtaining train predictions...")
    train_predictions = []
    train_targets = []
    for (i, m), target in usermovie2rating.items():
        # calculate the prediction for this movie
        prediction = predict(i, m)

        # save the prediction and target
        train_predictions.append(prediction)
        train_targets.append(target)

    test_predictions = []
    test_targets = []

    print("obtaining test predictions...")
    # same thing for test set
    for (i, m), target in usermovie2rating_test.items():
        # calculate the prediction for this movie
        prediction = predict(i, m)

        # save the prediction and target
        test_predictions.append(prediction)
        test_targets.append(target)

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))


if __name__ == "__main__":
    main()
