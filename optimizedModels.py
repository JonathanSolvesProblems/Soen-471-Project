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
from mlPrep import *

def gridsearch_rf_spark(parlerDataDirectory):
    from pyspark.ml.regression import RandomForestRegressor as spark_RFRegressor
    from pyspark.ml.tuning import CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import ParamGridBuilder

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

def gridsearch_random_forest_scikit(parlerDataDirectory):
    from sklearn.model_selection import GridSearchCV

    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }

    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

    train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
    print(train_x)
    print(train_y)

    grid_search.fit(train_features, train_labels)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, test_features, test_labels)
    
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

    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

    return val_y_pred, val_comparison, test_y_pred

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
    params = {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'log2', 'sqrt'], 'bootstrap': [True, False]}
    random = RandomizedSearchCV(rf, param_distributions=params, cv=5,
                                n_iter=10, verbose = 2, random_state = 0, return_train_score = True)
    random.fit(train_x, train_y)
    print('Best hyper parameters:', random.best_params_)

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

    return lr_model, predictions

gridsearch_rf_scikit(parlerDataDirectory)