from mlPrep import *
from pyspark.ml.regression import RandomForestRegressor as spark_RFRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.model_selection import RandomizedSearchCV

# 10:08 minutes took to run
def gridsearch_rf_spark(parlerDataDirectory):
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
    train, test = df.randomSplit([0.8, 0.2])
    cvModel = crossval.fit(train)
    predictions = cvModel.transform(test)
    # evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # rfPred = cvModel.transform(df)
    # rfResult = rfPred.toPandas()
    # plt.plot(rfResult.label, rfResult.prediction, 'bo')
    # plt.xlabel('Label')
    # plt.ylabel('Prediction')
    # plt.suptitle("Model Performance RMSE: %f" % rmse)
    # plt.show()
    predictions.show()
    
    return cvModel.bestModel.extractParamMap(), cvModel.bestModel, predictions

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

# 2:45
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

# gridsearch_rf_scikit(parlerDataDirectory)
# gridsearch_rf_spark(parlerDataDirectory)
# gridsearch_lg_spark(parlerDataDirectory)
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# model, val_y_pred, val_comparison, test_y_pred, test_comparison = linear_regression_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)

# plot_accuracy()

# plot_rmse_and_accuracy(["Test1", "Test2"], [10, 25])