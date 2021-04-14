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
    # test_y_pred_not_join = test_y_pred
    test_y_pred = pd.DataFrame(test_y_pred, columns=['test_predicted_upvotes']) 
    # test_comparison = test_y_pred.join(test_y, on = 'upvotes')
    # print(val_comparison)
    # print(test_y_pred)
    
    print(outputJson)
    test_y_pred.to_csv("test.csv", index = False, header = True)

    return val_y_pred, val_comparison, test_y_pred



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

def rf_spark_to_csv(predictions):
    
    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("probability", convert_array_to_string_udf(predictions["probability"]))
    predictions = predictions.withColumn("rawPrediction", convert_array_to_string_udf(predictions["rawPrediction"]))
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))

    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions


# linear_regression_spark(parlerDataDirectory)
# random_forest_spark(parlerDataDirectory)
# linear_regression_scikit(parlerDataDirectory)
# random_forest_scikit(parlerDataDirectory)


# gridsearch_rf_scikit(parlerDataDirectory)
## WORKING
# predictions = random_forest_spark(parlerDataDirectory)
# random_forest_spark_metrics(predictions)

# lr_model, pred = linear_regression_spark(parlerDataDirectory)
# linear_regression_spark_metrics(lr_model, pred)

# HERE
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# val_y_pred, val_comparison, test_y_pred = random_forest_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)

# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# model, val_y_pred, val_comparison, test_y_pred, test_comparison = linear_regression_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)

#gridsearch_rf_spark(parlerDataDirectory)

# gridsearch_lg_spark(parlerDataDirectory)

# HYPER Random Forest Scikit Learn
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# val_y_pred, val_comparison, test_y_pred = gridsearch_random_forest_scikit(parlerDataDirectory)
# scikit_metrics(test_y, test_y_pred)