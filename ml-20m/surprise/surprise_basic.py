from surprise import SVD
from surprise import Dataset, dump
from surprise.model_selection import cross_validate


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

'''
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
RMSE (testset)    0.9405  0.9295  0.9366  0.9341  0.9399  0.9361  0.0041
MAE (testset)     0.7409  0.7323  0.7379  0.7355  0.7437  0.7381  0.0040
Fit time          10.03   20.52   21.43   13.08   10.68   15.15   4.87
Test time         0.93    1.23    0.77    0.64    0.53    0.82    0.24
'''
