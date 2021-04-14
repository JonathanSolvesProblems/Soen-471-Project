from preprocess import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, when, concat_ws
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from sklearn import metrics
import numpy as np
from plot import *
from metrics import *

parlerDataDirectory = './parler_data000000000037.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parler-data/'

#SPARK MODELS
def prep_data_spark(parlerDataDirectory):
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData()
    # df = df.drop('body', 'createdAtformatted')
    # spark = init_spark()
    # df = spark.read.csv(parlerDataDirectory, header=True)
    df.show()
    # df.where(col("upvotes").isNull()).show()
    # df = df.filter(df.upvotes.isNotNull())
    df = df.dropna()
    # df = df.na.drop()
    df = df.withColumn("comments", col("comments").cast(FloatType())).\
        withColumn("followers", col("followers").cast(FloatType())).\
        withColumn("following", col("following").cast(FloatType())).\
        withColumn("impressions", col("impressions").cast(FloatType())).\
        withColumn("reposts", col("reposts").cast(FloatType())).\
        withColumn("verified", col("verified").cast(FloatType())). \
        withColumn("categoryIndexMimeType", col("categoryIndexMimeType").cast(FloatType())). \
        withColumn("categoryIndexDomains", col("categoryIndexDomains").cast(FloatType())). \
        withColumn("sentiment_score", col("sentiment_score").cast(FloatType())). \
        withColumn("hashtag significance", col("hashtag significance").cast(FloatType())). \
        withColumn("body significance", col("body significance").cast(FloatType())). \
        withColumn("upvotes", col("upvotes").cast(FloatType()))

    indexer = StringIndexer(inputCol='upvotes', outputCol='label')
    indexed = indexer.fit(df).transform(df)

    assembler = VectorAssembler(
        inputCols=['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                   'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance'],
        outputCol="features")
    output = assembler.transform(indexed)
    output.show()

    # stages = [indexer]
    # stages += [assembler]
    # cols = df.columns
    # pipeline = Pipeline(stages=stages)
    # pipelineModel = pipeline.fit(df)
    # df = pipelineModel.transform(df)
    # selectedCols = ['label', 'features'] + cols
    # df = df.select(selectedCols)
    return output

def prep_data_scikit(parlerDataDirectory):
    # df = pd.read_csv('part-00000-fe953ba9-c2ae-401a-bf98-58cbafc0ddcc-c000.csv')
    # df = df.drop('body')
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData().toPandas()
    # df = df.dropna(subset = ["upvotes"]) ``
    df = df.dropna() # may be removing more than we want

    # df.show()
    train, val = train_test_split(df, test_size=0.3)
    train, test = train_test_split(train, test_size=0.1)

    # Split the data into x and y
    train_x = train[
        ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
         'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']]
    train_y = train['upvotes']
    val_x = val[['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                 'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']]
    val_y = val['upvotes']
    test_x = test[['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                   'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']]
    test_y = test['upvotes']
    return train_x, train_y, val_x, val_y, test_x, test_y

def convert_array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'

def format_for_csv(predictions):
    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))
    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions

