from preprocess import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor as spark_RFRegressor
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from pyspark.ml.tuning import ParamGridBuilder

parlerDataDirectory = './parler_small.ndjson/'
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
    #val_comparison = val_y_pred.join(val_y)
    #print(val_comparison)

    # Make predictions using the test set
    test_y_pred = model.predict(test_x)
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes'])
    #test_comparison = test_y_pred.join(test_y)
    #print(test_comparison)

    return model, val_y_pred, test_y_pred

# Parameters to consider: n_estimators, criterion, max_depth, min_samples_split
def random_forest_scikit(parlerDataDirectory):
    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)

    rf = RandomForestRegressor()
    rf.fit(train_x, train_y)
    val_y_pred = rf.predict(val_x)
    val_y_pred = pd.DataFrame(val_y_pred, columns=['val_predicted_upvotes'])
    #val_comparison = val_y_pred.join(val_y)
    #print(val_comparison)

    # Make predictions using the test set
    test_y_pred = rf.predict(test_x)
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes'])
    #test_comparison = test_y_pred.join(test_y)
    #print(val_comparison)

    return val_y_pred, test_y_pred

#SPARK MODELS
def prep_data_spark(parlerDataDirectory):
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    df = preprocessor.getProcessedData()
    df = df.drop('body', 'createdAtformatted')
    # spark = init_spark()
    # df = spark.read.csv(parlerDataDirectory, header=True)
    df.show()
    #df = df.filter(df.upvotes.isNotNull())
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

    return output

def random_forest_classification_spark(parlerDataDirectory):

    df = prep_data_spark(parlerDataDirectory)
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    rf_model = rf.fit(df)
    predictions = rf_model.transform(test)
    predictions.show()
    return rf_model, predictions

def linear_regression_spark(parlerDataDirectory):
    df = prep_data_spark(parlerDataDirectory)
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    lr = LinearRegression(featuresCol='features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train)
    predictions = lr_model.transform(test)
    predictions.show()
    test_result = lr_model.evaluate(test)

    return lr_model, predictions

def random_forest_regression_spark(parlerDataDirectory):
    df = prep_data_spark(parlerDataDirectory)

    train, test = df.randomSplit([0.7, 0.3])
    rf = spark_RFRegressor(featuresCol="features")
    rf_model = rf.fit(df)
    predictions = rf_model.transform(test)

    predictions.select("prediction", "label", "features").show()
    return predictions

def random_forest_spark_metrics(predictions):
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)
    print("Test Error = %g" % (1.0 - accuracy))

    # Compute raw scores on the test set
    predictionAndLabels = predictions.select(['prediction', 'label']).rdd

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    confusionMatrix = metrics.confusionMatrix().toArray()
    print("Confusion Matrix\n %s" % confusionMatrix)

    # Overall statistics
    for i in predictions.select('label').distinct().collect():
        precision = metrics.precision(i)
        recall = metrics.recall(i)
        f1Score = metrics.fMeasure(i)
        print("Summary Stats")
        print("Precision = %s" % precision)
        print("Recall = %s" % recall)
        print("F1 Score = %s" % f1Score)

    # Statistics by class
    labels = predictions.select('prediction').distinct().collect()
    for label in sorted(labels):
        #print(label)
        print("Class %s precision = %s" % (label.prediction, metrics.precision(label.prediction)))
        print("Class %s recall = %s" % (label.prediction, metrics.recall(label.prediction)))
        print("Class %s F1 Measure = %s" % (label.prediction, metrics.fMeasure(label.prediction, beta=1.0)))

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
    #predictions = model.predict(test_data.map(lambda x: x.features))
    #labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
    acc = predictionAndLabels.filter(lambda x: x[0] == x[1]).count() / float(predictions.count())
    print('Accuracy %.3f' %acc)
    print("Model accuracy: %.3f%%" % (acc * 100))
    # Instantiate metrics object
    metrics = RegressionMetrics(predictionAndLabels)

    print("Mean Squared Error = %s" % metrics.meanSquaredError)
    print("Root Mean Squared Error = %s" % metrics.rootMeanSquaredError)
    print("R-squared = %s" % metrics.r2)
    print("Mean Absolute Error = %s" % metrics.meanAbsoluteError)
    print("Explained variance = %s" % metrics.explainedVariance)

def scikit_metrics(y_test, y_pred):
    print(y_test.compare(y_pred))
    print(y_test.count())
    y_test = y_test.to_numpy()
    y_pred = y_pred.to_numpy()
    print('Accuracy: ', np.count_nonzero(y_test == y_pred)/len(y_test))
    #print(len(diff)/y_test.count())
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    # print('Recall: ', metrics.recall_score)
    # print('Precision: ', metrics.precision_score)
    # print('F1 Measure: ', metrics.f1_score)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R-Squared Error:', metrics.r2_score(y_test, y_pred))
    print('Explained Variance Error:', metrics.explained_variance_score(y_test, y_pred))

def gridsearch_rf_spark(parlerDataDirectory):
    from pyspark.ml.regression import RandomForestRegressor as spark_RFRegressor
    from pyspark.ml.tuning import CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    df = prep_data_spark(parlerDataDirectory)

    rf = spark_RFRegressor(labelCol="label", featuresCol="features")
    #pipeline = Pipeline(stages=[df, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [2, 5, 10, 15, 20])\
        .addGrid(rf.maxBins, [5, 10, 20, 30])\
        .addGrid(rf.numTrees, [5, 20, 50, 100])\
        .build()
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

    crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid,
                              evaluator=evaluator, numFolds=2)
    (trainingData, testData) = df.randomSplit([0.8, 0.2])
    cvModel = crossval.fit(trainingData)
    predictions = cvModel.transform(testData)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    rfPred = cvModel.transform(df)
    rfResult = rfPred.toPandas()
    plt.plot(rfResult.label, rfResult.prediction, 'bo')
    plt.xlabel('Label')
    plt.ylabel('Prediction')
    plt.suptitle("Model Performance RMSE: %f" % rmse)
    plt.show()

    feature_list = ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                   'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']
    bestPipeline = cvModel.bestModel
    bestModel = bestPipeline.stages[1]
    importances = bestModel.featureImportances
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')

def gridsearch_lg_spark(parlerDataDirectory):
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator

    df = prep_data_spark(parlerDataDirectory)
    train, test = df.randomSplit([0.8, 0.2])

    # Create initial LinearRegression model
    lr = LinearRegression(labelCol="label", featuresCol="features")

    # Create ParamGrid for Cross Validation
    lrparamGrid = (ParamGridBuilder()
                   .addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
                   .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
                   .addGrid(lr.maxIter, [1, 5, 10, 20, 50])
                   .build())

    # Evaluate model
    lrevaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")

    # Create 5-fold CrossValidator
    lrcv = CrossValidator(estimator=lr, estimatorParamMaps=lrparamGrid, evaluator=lrevaluator, numFolds=5)

    # Run cross validations
    lrcvModel = lrcv.fit(train)
    print(lrcvModel)

    # Get Model Summary Statistics
    lrcvSummary = lrcvModel.bestModel.summary
    print("Coefficient Standard Errors: " + str(lrcvSummary.coefficientStandardErrors))
    print("P Values: " + str(lrcvSummary.pValues))  # Last element is the intercept

    # Use test set here so we can measure the accuracy of our model on new data
    lrpredictions = lrcvModel.transform(test)

    # cvModel uses the best model found from the Cross Validation
    # Evaluate best model
    print('RMSE:', lrevaluator.evaluate(lrpredictions))

def gridsearch_rf_scikit(parlerDataDirectory):
    from sklearn.model_selection import RandomizedSearchCV

    # parameters = {"max_depth": [3, 5, 10, ],
    #               "max_features": [1, 3, 5, 10],
    #               "min_samples_split": [1, 3, 10],
    #               "min_samples_leaf": [1, 3, 10],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"],
    #               "n_estimators": [10, 50, 100, 200]}
    # Run RandomizedSearchCV to tune the hyper-parameter
    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)

    rf = RandomForestRegressor()
    params = {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'log2', 'sqrt'], 'bootstrap': [True, False], "criterion": ["gini", "entropy"]}
    random = RandomizedSearchCV(rf, param_distributions=params, cv=5,
                                n_iter=5, verbose = 2, random_state = 42, return_train_score = True)
    random.fit(train_x, train_y)
    print('Best hyper parameters:', random.best_params_)

#predictions = random_forest_regression_spark(parlerDataDirectory)
# lr_model, pred = linear_regression_spark(parlerDataDirectory)
# linear_regression_spark_metrics(lr_model, pred)
# rf, predictions = random_forest_spark(parlerDataDirectory)
#random_forest_spark_metrics(predictions)
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# _, _, test_y_pred = linear_regression_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)
# random_forest_scikit(parlerDataDirectory)
gridsearch_rf_spark(parlerDataDirectory)