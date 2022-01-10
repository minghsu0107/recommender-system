import numpy as np
import pandas as pd
from surprise import SVDpp, Reader, Dataset, BaselineOnly
from surprise import accuracy, dump, model_selection
from surprise.model_selection import KFold


print("preparing data")
df = pd.read_csv('/tmp2/b07902053/ml-20m/small_rating.csv')
lower_rating = df['rating'].min()
upper_rating = df['rating'].max()
print(f'review range: {lower_rating} to {upper_rating}')


reader = Reader(rating_scale=(lower_rating, upper_rating))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)


# Algorithm predicting a random rating based on the distribution of the training set,
# which is assumed to be normal.
# cross_validate(BaselineOnly(), data, verbose=True)


algo = SVDpp(verbose=True)

param_grid = {'lr_all': [0.001, 0.01], 'reg_all': [0.1, 0.5]}
gs = model_selection.GridSearchCV(
    SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
print(gs.best_params['rmse'])

# trainset = data.build_full_trainset() # load full training data into memory
# algo.fit(trainset)

# define a cross-validation iterator
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)


dump.dump("svdpp.algo", algo, verbose=1)
algo, _ = dump.load("svdpp.algo")


print("make prediction")
uid = str(603)
iid = str(1037)
pred = algo.predict(uid=uid, iid=iid)
score = pred.est
print(score)

# get a list of all movie ids
iids = df['movieId'].unique()
# get a list of all movies user 50 has rated
iids50 = df.lod[df['userId'] == 50, 'movieId']
iids_to_pred = np.setdiff1d(iids, iids50)

# We'll just arbitrarily set all the ratings of this test set to 4, as they are not needed
testset = [[50, iid, 4] for iid in iids_to_pred]
predictions = algo.test(testset)
pred_ratings = np.array([pred.est for pred in predictions])
i_max = pred_ratings.argmax()
print(
    f'Top item for user 50 has movieId {iids_to_pred[i_max]} with predicted rating {pred_ratings[i_max]}')
