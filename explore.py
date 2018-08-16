import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Create a SparkSession and get the SparkContext from it.
# We can also get a SparkContext from using SparkContext("local", "app_name")
spark = SparkSession.builder.master("local").appName("first_app").getOrCreate()
sc = spark.sparkContext

# Basic example: Read in a file name and perform basic filter operations.
filename = 'file.txt'

logData = sc.textFile(filename).cache() # Read and cache the textFile

num_a = logData.filter(lambda x: 'a' in x).count() # Count how many lines 'a' in them
num_b = logData.filter(lambda x: 'b' in x).count() # Note filter applies to each line

print("num_a: {}\nnum_b: {}".format(num_a, num_b)) # Print the result

# Example of creating an RDD (resilient distributed dataset)
words = sc.parallelize (
   [
   "strawberry", 
   "lemon", 
   "pistachio",
   "earl grey", 
   "snickers",
   "creme brulee" 
   ]
)

counts = words.count() # Again with the counting function
print("Number of lines: {}".format(counts))




rdd = sc.parallelize(range(1, 1000)).map(lambda x: x * 10)
rdd.filter(lambda x: x % 100 == 0).first()
rdd.filter(lambda x: x % 100 == 0).take(5)
rdd.filter(lambda x: x % 100 == 0).collect()

rdd.map(lambda x: x*x).sum()
rdd.map(lambda x: x*x).max()
rdd.map(lambda x: x*x).min()

peopleRDD = sc.textFile('./data/people.txt')
peopleRDD = peopleRDD.map(lambda x: x.split('\t')) # Split tab-delimited file

peopleRDD.map(lambda t: (t[1], 1)) # Create some tuples that spark will see as (Key, Value)
peopleRDD.map(lambda t: (t[1], 1)).reduceByKey(lambda x,y: x + y).collect() # reduceByKey groups by first element (key)

peopleRDD.map(lambda t: (t[3], int(t[2])))
peopleRDD.map(lambda t: (t[3], int(t[2]))).reduceByKey(lambda x,y: x + y).collect()


import person # see person.py
from person import Person

peopleRDD = sc.textFile('./data/people.txt/').map(lambda x: Person().parse(x))

# Count people by gender
peopleRDD.map(lambda t: (t.gender, 1)).reduceByKey(lambda x, y: x + y).collect()

# Get youngest per gender
peopleRDD.map(lambda t: (t.gender, t.age)).reduceByKey(lambda x, y: min(x, y)).collect()


# Can read * as bash-like expressions 
salesRDD = sc.textFile('./data/sales_*.txt').map(lambda x: x.split('\t'))

states = [
("AL", "Alabama"),
("AK", "Alaska"),
("AR", "Arizona"),
("CA", "California")
]


populations = [
("AL", 4779736),
("AK", 710231),
("AR", 6392017)
]

states_rdd = sc.parallelize(states)
populations_rdd = sc.parallelize(populations)


states_rdd.join(populations_rdd)
populations_rdd.join(states_rdd, how="fullOuterJoin")

# Usually you want appname and master?
conf = SparkConf().setAppName(appName).setMaster(master) # master of cluster?

# If no need for spark conf, then just get spark sc
sc = SparkContext()


# Using SQLContext...
sc = SparkContext()
sqlCtx = SQLContext(sc)

people_df = sqlCtx.read.json("./data/people.json") # will create a database which takes a while

# When we query from sqlCtx, there's a constant query optimization overhead
# which can seem slow for our small data

# We can register this dataframe as a table for our SQLContext
# And we can then run SQL queries from the SQLContext.

# Register temp table
people_df.registerTempTable("people")

sqlCtx.sql("SELECT name, age FROM people").show()
sqlCtx.sql("SELECT gender, AVG(age) FROM people GROUP BY gender").show()


from pyspark.sql.types import StructField, StringType, IntegerType, StructType

sales_rdd = sc.textFile('./data/sales_*.txt').map(lambda x: x.split('\t'))\
    .map(lambda t: (t[0], t[1], t[2], int(t[3])))

sales_fields = [
    StructField('day', StringType(), False), # name, type, nullable
    StructField('store', StringType(), False),
    StructField('product', StringType(), False),
    StructField('quantity', IntegerType(), False)
]

sales_schema = StructType(sales_fields)
sales = sqlCtx.createDataFrame(sales_rdd, sales_schema) # Create dataframe from rdd

sqlCtx.registerDataFrameAsTable(sales, "sales") # another way to add DataFrame to our SQLContext tables


sqlCtx.sql("SELECT * FROM sales").show()
sqlCtx.sql("SELECT * FROM people").explain()


#######################
#       PySpark       #
#######################
# GraphX # MLlib # SQL# # What is GraphX and Mlib???
#######################
#        Spark        #
#######################


# There's a driver program, and it wakes up these worker nodes which are close to the data
# and will perform some task (delegated by driver program)

# Transformations
# map, flatmap, filter, distinct, sample, union, intersection, subtract, cartesian

# Actions
# collect, count, take, takeOrdered, reduce, aggregate, foreach


from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

df = sqlCtx.read.csv('./data/churn.all')

numeric_cols

# Get DataFrame from SQL
df_adult = spark.table("adult")

result = spark.sql(
    """
    SELECT
        *
    FROM
        adult
    """)

df_adult.printSchema() # Look at the df schema

from pyspark.sql.functions import when, col, mean, desc, round

df_result = df_adult.select(
    df_adult['occupation']
    when( col('marital_status') == ' Divorced', 1).otherwise(0).alias('is_divorced')
)

result.show()


# DataFrame syntax SQL
df_result = df_adult.select(
  df_adult['education_num'].alias('education'),
  when( df_adult['marital_status'] == ' Never-married', 1).otherwise(0).alias('bachelor')
)
df_result = df_result.groupBy('education').agg(
  round(mean('bachelor'), 2).alias('bachelor_rate'),
  round(mean('bachelor'), 2).alias('bachelor_rate2') 
)
df_result = df_result.orderBy(desc('bachelor_rate'))
df_result.show(1)