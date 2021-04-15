from mlPrep import *
from optimizedModels import *
from pyspark.ml.regression import RandomForestRegressor as spark_RFRegressor

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
    print("******************************", train_x)
    print("******************************", train_y)

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

    feature_list = ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                    'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']

    importances = rf.feature_importances_
    # importances = random.feature_importances_

    # print("**************", importances)
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    plt.show()

    return val_y_pred, val_comparison, test_y_pred

def random_forest_spark(parlerDataDirectory):

    df = prep_data_spark(parlerDataDirectory)

    df.show()
    train, val, test = df.randomSplit([0.6, 0.2, 0.2])

    #featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

    rf = spark_RFRegressor(labelCol="label", featuresCol="features")
    rf_model = rf.fit(df)
    print(rf_model.featureImportances)
    predictions = rf_model.transform(test)
    predictions.show()

    importances = rf_model.featureImportances

    feature_list = ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                    'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']

    print("importace*****", importances)
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    # plt.show()

    return predictions

def rf_spark_to_csv(predictions):
    
    convert_array_to_string_udf = udf(convert_array_to_string, StringType())
    predictions = predictions.withColumn("probability", convert_array_to_string_udf(predictions["probability"]))
    predictions = predictions.withColumn("rawPrediction", convert_array_to_string_udf(predictions["rawPrediction"]))
    predictions = predictions.withColumn("features", convert_array_to_string_udf(predictions["features"]))

    predictions.coalesce(1).write.format("csv").save(outputJson, header = True)

    return predictions


def get_features_spark(model):
    feature_list = ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                    'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']
    importances = model.featureImportances
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    plt.show()

def get_features_scikit(model):

    feature_list = ['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType',
                    'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance']
    importances = model.feature_importances_
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    plt.show()


# gridsearch_rf_scikit(parlerDataDirectory)
# print(params)
# get_features_scikit(model)

# gridsearch_rf_spark(parlerDataDirectory)

# spark_params, spark_model, pred = gridsearch_rf_spark(parlerDataDirectory)
# print(spark_params)
# get_features_spark(spark_model)

# gridsearch_rf_scikit(parlerDataDirectory)

# t, model = gridsearch_rf_scikit(parlerDataDirectory)
# get_features_scikit(model)


train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
model, val_y_pred, val_comparison, test_y_pred, test_comparison = linear_regression_scikit(parlerDataDirectory)
visualize_linear_regression_scikit(test_x, test_y, test_y_pred)

# linear_regression_spark(parlerDataDirectory)
# random_forest_spark(parlerDataDirectory)
# linear_regression_scikit(parlerDataDirectory)
# random_forest_scikit(parlerDataDirectory)

# gridsearch_rf_scikit(parlerDataDirectory)
## WORKING

# HERE
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# val_y_pred, val_comparison, test_y_pred = random_forest_scikit(parlerDataDirectory)
# print(test_x.values.flatten())
# print(test_x.index[0].shape)
# print(test_y.shape)
# visualize_linear_regression_scikit(test_x.index[0], test_y.values, test_y_pred)
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


# Accuracy: 0.90
# RMSE: 1.78
# Time: 1:57
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# val_y_pred, val_comparison, test_y_pred = random_forest_scikit(parlerDataDirectory)
# sci_rf_acc, sci_rf_rmse = scikit_metrics(test_y, test_y_pred)

# Accuracy: 0
# 1.71
# Time: 1:52
# train_x, train_y, val_x, val_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
# model, val_y_pred, val_comparison, test_y_pred, test_comparison = linear_regression_scikit(parlerDataDirectory)
# sci_lr_acc, sci_lr_rmse = scikit_metrics(test_y, test_y_pred)

# Accuracy: 0.91
# RMSE 0.62
# Time: 5:04
# random_forest_spark(parlerDataDirectory)
# spark_rf_acc, spark_rf_rmse = random_forest_spark_metrics(predictions)

# 0.0
# 0.91
# Time: 4:27
# lr_model, pred = linear_regression_spark(parlerDataDirectory)
# spark_lr_acc, spark_lr_rmse = linear_regression_spark_metrics(lr_model, pred)
# plot_accuracy(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [10, 10, 10, spark_lr_acc])
# plot_rmse(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [10, 10, 10, spark_lr_rmse])

# plot_accuracy(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [0.91, 0.0, 0.90, 0])
# plot_rmse(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [0.62, 0.91, 1.78, 1.71])

# plot_accuracy(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [spark_rf_acc, spark_lr_acc, sci_rf_acc, sci_lr_acc])
# plot_rmse(["Spark RF", "Spark LR", "Sci RF", "Sci LR"], [spark_rf_rmse, spark_lr_rmse, sci_rf_rmse, sci_lr_rmse])

# plot_time(["Spark GridSearch RF", "Sci GridSearch RF", "Spark RF", "Sci RF", "Spark LR", "Sci LR"], [10.08, 2.45, 5.04, 1.57, 4.27, 1.52])