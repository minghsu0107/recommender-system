from pyspark import SparkConf
from pyspark.context import SparkContext

sc = SparkContext.getOrCreate(SparkConf())

pp = (('cat', 2), ('cat', 5), ('book', 4), ('cat', 12))
qq = (("cat", 2), ("cup", 5), ("mouse", 4), ("cat", 12))

pairRDD1 = sc.parallelize(pp)
pairRDD2 = sc.parallelize(qq)

print(pairRDD1.collect())
print(pairRDD2.collect())

print(pairRDD1.join(pairRDD2).collect())
'''
[('cat', (2, 2)), ('cat', (2, 12)), 
 ('cat', (5, 2)), ('cat', (5, 12)), 
 ('cat', (12, 2)), ('cat', (12, 12))]
'''

print(pairRDD1.leftOuterJoin(pairRDD2).collect())
'''
[('book', (4, None)), ('cat', (2, 2)), 
 ('cat', (2, 12)), ('cat', (5, 2)), 
 ('cat', (5, 12)), ('cat', (12, 2)), ('cat', (12, 12))]
'''

print(pairRDD1.rightOuterJoin(pairRDD2).collect())
'''
[('cat', (2, 2)), ('cat', (2, 12)), ('cat', (5, 2)), 
 ('cat', (5, 12)), ('cat', (12, 2)), ('cat', (12, 12)), 
 ('cup', (None, 5)), ('mouse', (None, 4))]
'''

print(pairRDD1.fullOuterJoin(pairRDD2).collect())
'''
[('book', (4, None)), ('cat', (2, 2)), 
 ('cat', (2, 12)), ('cat', (5, 2)), ('cat', (5, 12)), 
 ('cat', (12, 2)), ('cat', (12, 12)), ('cup', (None, 5)), ('mouse', (None, 4))]
'''

print(pairRDD1.union(pairRDD2).collect())
'''
[('cat', 2), ('cat', 5), ('book', 4), 
 ('cat', 12), ('cat', 2), ('cup', 5), ('mouse', 4), ('cat', 12)]
'''
