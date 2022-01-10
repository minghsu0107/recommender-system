from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

sc = SparkContext.getOrCreate(SparkConf())
spark = SQLContext(sc)

valuesA = [('Pirate', 1), ('Monkey', 2), ('Ninja', 3), ('Spaghetti', 4)]
TableA = spark.createDataFrame(valuesA, ['name', 'id'])

valuesB = [('Rutabaga', 1), ('Pirate', 2), ('Ninja', 3),
           ('Ninja', 4), ('Darth Vader', 5)]
TableB = spark.createDataFrame(valuesB, ['name', 'id'])

TableA.show()
TableB.show()

'''
+---------+---+
|     name| id|
+---------+---+
|   Pirate|  1|
|   Monkey|  2|
|    Ninja|  3|
|Spaghetti|  4|
+---------+---+

+-----------+---+
|       name| id|
+-----------+---+
|   Rutabaga|  1|
|     Pirate|  2|
|      Ninja|  3|
|      Ninja|  4|
|Darth Vader|  5|
+-----------+---+
'''

ta = TableA.alias('ta')
tb = TableB.alias('tb')

inner_join = ta.join(tb, ta.name == tb.name, how='inner')
inner_join.show()
'''
SELECT ta.*, tb.*
FROM ta
INNER JOIN tb
ON ta.name = tb.name

+------+---+------+---+
|  name| id|  name| id|
+------+---+------+---+
| Ninja|  3| Ninja|  3|
| Ninja|  3| Ninja|  4|
|Pirate|  1|Pirate|  2|
+------+---+------+---+
'''
inner_join = ta.join(tb, 'name', how='inner')
inner_join.show()
'''
+------+---+---+
|  name| id| id|
+------+---+---+
| Ninja|  3|  3|
| Ninja|  3|  4|
|Pirate|  1|  2|
+------+---+---+
'''
left_join = ta.join(tb, ta.name == tb.name,
                    how='left')  # Could also use 'left_outer'
left_join.show()
'''
+---------+---+------+----+
|     name| id|  name|  id|
+---------+---+------+----+
|Spaghetti|  4|  null|null|
|    Ninja|  3| Ninja|   3|
|    Ninja|  3| Ninja|   4|
|   Pirate|  1|Pirate|   2|
|   Monkey|  2|  null|null|
+---------+---+------+----+
'''

# Could also use 'right_outer'
right_join = ta.join(tb, ta.name == tb.name, how='right')
right_join.show()
'''
+------+----+-----------+---+
|  name|  id|       name| id|
+------+----+-----------+---+
|  null|null|   Rutabaga|  1|
| Ninja|   3|      Ninja|  3|
| Ninja|   3|      Ninja|  4|
|Pirate|   1|     Pirate|  2|
|  null|null|Darth Vader|  5|
+------+----+-----------+---+
'''
full_outer_join = ta.join(tb, ta.name == tb.name,
                          how='full')  # Could also use 'full_outer'
full_outer_join.show()
'''
+---------+----+-----------+----+
|     name|  id|       name|  id|
+---------+----+-----------+----+
|     null|null|   Rutabaga|   1|
|Spaghetti|   4|       null|null|
|    Ninja|   3|      Ninja|   3|
|    Ninja|   3|      Ninja|   4|
|   Pirate|   1|     Pirate|   2|
|   Monkey|   2|       null|null|
|     null|null|Darth Vader|   5|
+---------+----+-----------+----+
'''
rdd = sc.parallelize([(1, 'Alice', 18), (2, 'Andy', 19),
                      (3, 'Bob', 17), (4, 'Justin', 21), (5, 'Cindy', 20)])
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])
df = spark.createDataFrame(rdd, schema)
df.show()
'''
+---+------+---+
| id|  name|age|
+---+------+---+
|  1| Alice| 18|
|  2|  Andy| 19|
|  3|   Bob| 17|
|  4|Justin| 21|
|  5| Cindy| 20|
+---+------+---+
'''
rdd2 = sc.parallelize([('Alice', 160), ('Andy', 159),
                       ('Bob', 170), ('Cindy', 165), ('Rose', 160)])
schema2 = StructType([
    StructField("name", StringType(), True),
    StructField("height", IntegerType(), True)
])
df2 = spark.createDataFrame(rdd2, schema2)
df2.show()
# df2.show(1) # show top 1 row
df2.describe().show()
'''
+-----+------+
| name|height|
+-----+------+
|Alice|   160|
| Andy|   159|
|  Bob|   170|
|Cindy|   165|
| Rose|   160|
+-----+------+

+-------+-----+-----------------+
|summary| name|           height|
+-------+-----+-----------------+
|  count|    5|                5|
|   mean| null|            162.8|
| stddev| null|4.658325879540846|
|    min|Alice|              159|
|    max| Rose|              170|
+-------+-----+-----------------+
'''
inner_join2 = df.join(df2, "name", "inner").select(
    "id", df.name, "age", "height").orderBy("id")
'''
+---+-----+---+------+
| id| name|age|height|
+---+-----+---+------+
|  1|Alice| 18|   160|
|  2| Andy| 19|   159|
|  3|  Bob| 17|   170|
|  5|Cindy| 20|   165|
+---+-----+---+------+
'''
print(inner_join2.collect()[0].name)  # Alice
print(inner_join2.columns)  # ['id', 'name', 'age', 'height']
# [('id', 'int'), ('name', 'string'), ('age', 'int'), ('height', 'int')]
print(inner_join2.dtypes)
df.filter(df.age == 20).show()

df.createOrReplaceTempView("people")
sql_results = spark.sql("SELECT * FROM people")  # sql_results is a dataframe
sql_results.show()


### Column Operations ###
columns_to_drop = ['id', 'name']
df.drop(*columns_to_drop).show()

df.withColumn('doubleage', df['age']*2).show()
'''
+---+------+---+---------+
| id|  name|age|doubleage|
+---+------+---+---------+
|  1| Alice| 18|       36|
|  2|  Andy| 19|       38|
|  3|   Bob| 17|       34|
|  4|Justin| 21|       42|
|  5| Cindy| 20|       40|
+---+------+---+---------+
'''
df.withColumn('newage', df['age']).show()  # create new column

print(df.select("name").collect())  # returns a list of Row objects
name_list = [row.name for row in df.select("name").collect()]

### Row Operations ###
print(df.filter(df.age == 20).count())
result = df.filter(df.age == 20).collect()  # rows that satisfy the condition
print(result[0][0], result[0][1], result[0][2])  # 5 Cindy 20

for item in result[0].asDict().items():
    print(item)
'''
('id', 5)
('name', 'Cindy')
('age', 20)
'''
