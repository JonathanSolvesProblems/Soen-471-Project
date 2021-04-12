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


parlerDataDirectory = './parler_data000000000037.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parler-data/'

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    # conf = SparkConf().setAppName("test").setMaster("local")
    # sc = SparkContext(conf=conf)
    # sc.setLogLevel("ERROR")
    return spark
    
def convert_array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'

def format_for_csv(predictions):
    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))
    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions

#SCIKIT MODELS
def prep_data_scikit(parlerDataDirectory):
    # df = pd.read_csv('part-00000-fe953ba9-c2ae-401a-bf98-58cbafc0ddcc-c000.csv')
    # df = df.drop('body')
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData().toPandas()
    # df = df.dropna(subset = ["upvotes"]) 
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

    test_y_pred.to_csv("test.csv", index = False, header = True)

    return model, val_y_pred, val_comparison, test_y_pred, test_comparison

# Parameters to consider: n_estimators, criterion, max_depth, min_samples_split
def random_forest_scikit(parlerDataDirectory):
    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
    print(train_x)
    print(train_y)

    rf = RandomForestRegressor()
    rf.fit(train_x, train_y)
    val_y_pred = rf.predict(val_x)
    val_y_pred = pd.DataFrame(val_y_pred, columns=['val_predicted_upvotes'])
    val_comparison = val_y_pred.join(val_y)
    print(val_comparison)

    # Make predictions using the test set
    test_y_pred = rf.predict(test_x)
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes']) 
    # test_comparison = test_y_pred.join(test_y, on = 'upvotes')
    # print(val_comparison)
    # print(test_y_pred)
    
    print(outputJson)
    test_y_pred.to_csv("test.csv", index = False, header = True)

    return val_y_pred, val_comparison, test_y_pred

#SPARK MODELS
def prep_data_spark(parlerDataDirectory):
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData()
    # df = df.drop('body', 'createdAtformatted')
    spark = init_spark()
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

def random_forest_spark(parlerDataDirectory):

    df = prep_data_spark(parlerDataDirectory)

    df.show()
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    #featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(df)
    predictions = rf_model.transform(test)
    predictions.show()

    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("probability", convert_array_to_string_udf(predictions["probability"]))
    predictions = predictions.withColumn("rawPrediction", convert_array_to_string_udf(predictions["rawPrediction"]))
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))

    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions

def linear_regression_spark(parlerDataDirectory):
    df = prep_data_spark(parlerDataDirectory)
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    lr = LinearRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    predictions = lr_model.transform(test)
    predictions.show()

    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))
    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions

# linear_regression_spark(parlerDataDirectory)
# random_forest_spark(parlerDataDirectory)
# linear_regression_scikit(parlerDataDirectory)
# random_forest_scikit(parlerDataDirectory)



def random_forest_spark_metrics(predictions):
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")

    drop_columns = ["body", "comments", "createdAtformatted", "followers", "following", "impressions", "reposts", "sensitive", "upvotes", "verified",\
    "categoryIndexMimeType", "categoryIndexDomains", "sentiment_score", "hashtag significance", "body significance", "features", "rawPrediction", "probability"]

    predictions = predictions.drop(*drop_columns)

    predictions.show()

    accuracy = evaluator.evaluate(predictions)
    print(accuracy)
    print("Test Error = %g" % (1.0 - accuracy))

    # Compute raw scores on the test set
    predictionAndLabels = predictions.select(['prediction', 'label']).rdd

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)

    # Overall statistics
    confusionMatrix = metrics.confusionMatrix().toArray()
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Confusion Matrix\n %s" % confusionMatrix)
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    # Statistics by class
    labels = predictions.select('prediction').distinct().collect()
    for label in sorted(labels):
        print(label)
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

def linear_regression_spark_metrics(lr_model, predictions):
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    predictionAndLabels = predictions.select(['prediction', 'label']).rdd

    # Instantiate metrics object
    metrics = RegressionMetrics(predictionAndLabels)

    print("Mean Squared Error = %s" % metrics.meanSquaredError)
    print("Root Mean Squared Error = %s" % metrics.rootMeanSquaredError)
    print("R-squared = %s" % metrics.r2)
    print("Mean Absolute Error = %s" % metrics.meanAbsoluteError)
    print("Explained variance = %s" % metrics.explainedVariance)

def scikit_metrics(y_test, y_pred):

    print('Accuracy: ', metrics.accuracy_score)
    print('Recall: ', metrics.recall_score)
    print('Precision: ', metrics.precision_score)
    print('F1 Measure: ', metrics.f1_score)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R-Squared Error:', metrics.r2_score(y_test, y_pred))
    print('Explained Variance Error:', metrics.explained_variance_score(y_test, y_pred))

#lr_model, pred = linear_regression_spark(parlerDataDirectory)
#linear_regression_spark_metrics(lr_model, pred)

# rf, 
predictions = random_forest_spark(parlerDataDirectory)
random_forest_spark_metrics(predictions)

# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# _, _, test_y_pred = linear_regression_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)
#random_forest_scikit(parlerDataDirectory)