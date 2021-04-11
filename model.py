from preprocess import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

parlerDataDirectory = './part-00000-5f5fb812-5c87-4879-895b-09203ca5e852-c000.csv/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # conf = SparkConf().setAppName("test").setMaster("local")
    # sc = SparkContext(conf=conf)
    return spark


#SCIKIT MODELS
def prep_data_scikit(parlerDataDirectory):
    # df = pd.read_csv('part-00000-fe953ba9-c2ae-401a-bf98-58cbafc0ddcc-c000.csv')
    # df = df.drop('body')
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData().toPandas()
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

def linear_regression_scikit(parlerDataDirectory):
    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)

    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)

    # Make predictions using the validation set
    val_y_pred = model.predict(val_x)
    val_y_pred = pd.DataFrame(val_y_pred, columns=['val_predicted_upvotes'])
    val_comparison = val_y_pred.join(val_y)
    print(val_comparison)

    # Make predictions using the test set
    test_y_pred = model.predict(test_x)
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes'])
    test_comparison = test_y_pred.join(test_y)
    print(test_comparison)

    return model, val_y_pred, val_comparison, test_y_pred, test_comparison

# Parameters to consider: n_estimators, criterion, max_depth, min_samples_split
def random_forest_scikit(parlerDataDirectory):
    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)

    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    val_y_pred = rf.predict(val_x)
    val_y_pred = pd.DataFrame(val_y_pred, columns=['val_predicted_upvotes'])
    val_comparison = val_y_pred.join(val_y)
    print(val_comparison)

    # Make predictions using the test set
    test_y_pred = rf.predict(test_x)
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes'])
    test_comparison = test_y_pred.join(test_y)
    print(val_comparison)

    return val_y_pred, val_comparison, test_y_pred, test_comparison

#SPARK MODELS
def prep_data_spark(parlerDataDirectory):
    # preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    # preprocessor.preprocessJson(parlerDataDirectory)
    # df = preprocessor.getProcessedData()
    # df = df.drop('body', 'createdAtformatted')
    spark = init_spark()
    df = spark.read.csv(parlerDataDirectory, header=True)
    df.show()
    #df = df.filter(df.upvotes.isNotNull())
    df = df.na.drop()
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
    #output.show()

    # stages = [indexer]
    # stages += [assembler]
    # cols = df.columns
    # pipeline = Pipeline(stages=stages)
    # pipelineModel = pipeline.fit(df)
    # df = pipelineModel.transform(df)
    # selectedCols = ['label', 'features'] + cols
    # df = df.select(selectedCols)
    return output

def random_forest_spark(parlerDataDirectory):

    df = prep_data_spark(parlerDataDirectory)

    df.show()
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    #featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(df)
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
random_forest_spark(parlerDataDirectory)
#linear_regression_scikit(parlerDataDirectory)
#random_forest_scikit(parlerDataDirectory)
