from preprocess import *
#from plot import *
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from tfidf import *
from pyspark.sql.functions import col, when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'
upvoteBins = 4

def prep_data_spark(parlerDataDirectory):
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData()
    df = df.drop('body', 'createdAtformatted')

    label_stringIdx = StringIndexer(inputCol='upvotes', outputCol='label')
    stages = [label_stringIdx]
    assembler = VectorAssembler(
        inputCols=['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                   'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance'],
        outputCol="features")
    stages += [assembler]

    cols = df.columns
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)
    return df

def random_forest_spark(parlerDataDirectory):

    df = prep_data_spark(parlerDataDirectory)
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(train)
    predictions = rf_model.transform(test)
    predictions.show()
    return predictions

def linear_regression_spark(parlerDataDirectory):
    df = prep_data_spark(parlerDataDirectory)
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    lr = LinearRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

#linear_regression_spark(parlerDataDirectory)

