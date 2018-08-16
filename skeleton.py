# To run this on a cluster we can use:
# spark-submit --packages mysql:mysql-connector-java:5.1.38 --master local skeleton.py
# Read docs for spark-submit for more information about arguments and configs

from itertools import chain
import pandas as pd

import pyspark
from pyspark import SparkContext

from pyspark.sql.functions import when, col, mean, desc, round
from pyspark.sql.types import * # Import types for SQL

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


spark = SparkSession.builder.master("local").appName("skeleton").getOrCreate()

sqlContext = SQLContext(spark)

adults_df = sqlContext.read.format("jdbc").options(
                url ="jdbc:mysql://localhost/pyspark", # Location of the SQL server
                driver="com.mysql.jdbc.Driver",
                dbtable="adults",
                user="root",
                password="password"
            ).load()

# We can register the DataFrame as a table if we want to run spark.sql queries on it
sqlContext.registerDataFrameAsTable(adults_df, "adults")

# Running some sql queries on our table registered from the DataFrame
spark.sql(
    """
    SELECT 
        occupation,
        marital_status,
        IF(marital_status == 'Divorced', 1, 0) as is_divorced 
    FROM 
        adults
    """).show(10)

# We can also just work with the DataFrame if we want to use DataFrame syntax
result = adults_df.select(
        adults_df['occupation'],
        adults_df['marital_status'], 
        when(adults_df['marital_status'] == 'Divorced', 1).otherwise(0).alias("is_divorced")
    )
result.show(10)


# Create label column
adults_df = adults_df.withColumn('label', when(adults_df.income.like('>50K%'), 0).otherwise(1))
# adults_df.groupBy('income').count()

# Cleaning up features for modeling:
# Get column names from the DataFrame
columns = adults_df.columns

# Separate our categorical variables and numeric variables
categoricalCols = ["workclass", "education", "marital_status", 
    "occupation", "relationship", "race", "gender", "native_country"]

numericCols = ["age", "fnlwgt", "educational_num", "capital_gain", "capital_loss", "hours_per_week"]

stages = [] # Stages in the pipeline

# First, we need to turn the categorical variables into one-hot encodings. 
# We do this in two steps: StringIndexer and OneHotEncoderEstimator
for col in categoricalCols:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col + "_index")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[col + "_one_hot"])
    stages += [stringIndexer, encoder]

# Assemble all the columns into a single vector, called "features"
assemblerInputs = [c + "_one_hot" for c in categoricalCols] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

print(stages)

# Create a Pipeline.
pipeline = Pipeline(stages=stages)

# Fit pipeline to the dataset
pipelineModel = pipeline.fit(adults_df)

# Transform the dataset
dataset = pipelineModel.transform(adults_df)

dataset.select("label", "features").show(10) # (number of elements, [indices], [values at indicies])

# Split into train, test
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())


# Create LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# set threshold for the probability above which to predict a 1
# lr.setThreshold(training_data_positive_rate)
# lr.setThreshold(0.5) # could use this if knew you had balanced data

# Train model with Training Data
lrModel = lr.fit(trainingData)

# get training summary used for eval metrics and other params
lrTrainingSummary = lrModel.summary

# Make predictions on test data
lrPredictions = lrModel.transform(testData)

# Show predictions
lrPredictions.select("label", "prediction", "probability").show(10)



# Function to get a bunch of different metrics! 
def print_performance_metrics(predictions):
    # Evaluate model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
    print("auc = {}".format(auc))
    print("aupr = {}".format(aupr))

    # Get RDD of predictions and labels for eval metrics
    predictionAndLabels = predictions.select("prediction","label").rdd

    # Instantiate metrics objects
    binary_metrics = BinaryClassificationMetrics(predictionAndLabels)
    multi_metrics = MulticlassMetrics(predictionAndLabels)

    # Area under precision-recall curve
    print("Area under PR = {}".format(binary_metrics.areaUnderPR))
    # Area under ROC curve
    print("Area under ROC = {}".format(binary_metrics.areaUnderROC))
    # Accuracy
    print("Accuracy = {}".format(multi_metrics.accuracy))
    # Confusion Matrix
    print(multi_metrics.confusionMatrix())
    # F1
    print("F1 = {}".format(multi_metrics.fMeasure(1.0)))
    # Precision
    print("Precision = {}".format(multi_metrics.precision(1.0)))
    # Recall
    print("Recall = {}".format(multi_metrics.recall(1.0)))
    # FPR
    print("FPR = {}".format(multi_metrics.falsePositiveRate(1.0)))
    # TPR
    print("TPR = {}".format(multi_metrics.truePositiveRate(1.0)))


print_performance_metrics(lrPredictions)

# We can see parameters for the model
print(lr.explainParams())


# Create ParamGrid for Cross Validation
lrParamGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [2, 5])
             .build())

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# Create the crossvalidator with the estimator, the grid, and the evaluation metric
lrCv = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=evaluator, numFolds=2)
lrCvModel = lrCv.fit(trainingData)

# Look at best parameters
print(lrCvModel.bestModel._java_obj.getRegParam())
print(lrCvModel.bestModel._java_obj.getElasticNetParam())
print(lrCvModel.bestModel._java_obj.getMaxIter())

lrCvPredictions = lrCvModel.transform(testData)
lrCvPredictions.show()

# Show coefficients of each feature
attrs = sorted(
    (attr["idx"], attr["name"]) for attr in (chain(*lrCvPredictions
        .schema['features']
        .metadata["ml_attr"]["attrs"].values())))

lrCoeffDF = pd.DataFrame([(str(name), float(lrCvModel.bestModel.coefficients[idx])) for idx, name in attrs], columns=["feature", "coefficient"])
print(lrCoeffDF.sort_values(by=['coefficient'], ascending=False))


















