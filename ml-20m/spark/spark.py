from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import os

# increase memory
# SparkContext.setSystemProperty('spark.driver.memory', '10g')
# SparkContext.setSystemProperty('spark.executor.memory', '10g')

# standalone mode (client mode)
conf = SparkConf().setAppName("PySpark App")
sc = SparkContext.getOrCreate(conf=conf)

# load in the data
data = sc.textFile(name="/tmp2/b07902053/ml-20m/small_rating.csv")
# spark will automatically decompress
# data = sc.textFile(name="/tmp2/b07902053/ml-20m/small_rating.csv.gz")

# filter out header
header = data.first()  # extract header
data = data.filter(lambda row: row != header)

# convert into a sequence of Rating objects
ratings = data.map(
    lambda l: l.split(',')
).map(
    lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))
)

# split into train and test
train, test = ratings.randomSplit([0.8, 0.2])

# train the model
K = 10  # latent dimensionality
epochs = 10
model = ALS.train(train, K, epochs)


# save the model
# its a directory containing data and meta data
model.save(sc, "rec-spark.model")
# load the model
same_model = MatrixFactorizationModel.load(sc, "rec-spark.model")


# evaluate the model

# train
x = train.map(lambda p: (p[0], p[1]))
# for each predicted rating objects, transform to ((user_id, movie_id), prediction))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
# inner joins on first item by default: (user_id, movie_id)
# each row of result is: ((user_id, movie_id), (rating, prediction))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("train mse: %s" % mse)


# test
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
# print out the first 5 prediction
print(p.take(5))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("test mse: %s" % mse)
