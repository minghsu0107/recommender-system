from pyspark import SparkConf
from pyspark.context import SparkContext

sc = SparkContext.getOrCreate(SparkConf())

l = ["Hadoop", "Spark", "Hive", "Spark"]
rdd = sc.parallelize(l)

pairRDD = rdd.map(lambda word: (word, 1))
pairRDD.foreach(print)
'''
(Hadoop, 1)
(Spark, 1)
(Hive, 1)
(Spark, 1)
'''

# combine entries with a same key
pairRDD.reduceByKey(lambda a, b: a + b).foreach(print)
'''
(Spark, 2)
(Hive, 1)
(Hadoop, 1)
'''

# ex: ('spark',1), ('spark',2), ('hadoop',3), ('hadoop',5)
# -> ('spark',(1,2)), ('hadoop',(3,5))
pairRDD.groupByKey().foreach(print)
'''
('Spark', <pyspark.resultiterable.ResultIterable object at 0x7f23d8bff0a0>)
('Hadoop', <pyspark.resultiterable.ResultIterable object at 0x7f23d8bff0a0>)
('Hive', <pyspark.resultiterable.ResultIterable object at 0x7f23d8bff0a0>)
'''
print(pairRDD.keys().collect())  # ['Hadoop', 'Spark', 'Hive', 'Spark']
print(pairRDD.values().collect())  # [1, 1, 1, 1]

print(pairRDD.sortByKey().collect())
# [('Hadoop', 1), ('Hive', 1), ('Spark', 1), ('Spark', 1)]

print(pairRDD.sortByKey(False).collect())
# [('Spark', 1), ('Spark', 1), ('Hive', 1), ('Hadoop', 1)]

print(pairRDD.sortBy(lambda a: a[0], False).collect())
# [('Spark', 1), ('Spark', 1), ('Hive', 1), ('Hadoop', 1)]

print(pairRDD.mapValues(lambda x: x + 1).collect())
# [('Hadoop', 2), ('Spark', 2), ('Hive', 2), ('Spark', 2)]

# mixed example: calculate mean for each word
rdd = sc.parallelize([("spark", 2), ("hadoop", 6),
                      ("hadoop", 4), ("spark", 6)])
rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).\
    mapValues(lambda x: (x[0] / x[1])).collect()
# [('hadoop', 5.0), ('spark', 4.0)]
