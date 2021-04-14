from mlPrep import *

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
        i = i.label
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

    evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE): %.2f" % rmse)

    return accuracy, rmse

def linear_regression_spark_metrics(lr_model, predictions):
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    predictionAndLabels = predictions.select(['prediction', 'label']).rdd
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)
    print("Test Error = %g" % (1.0 - accuracy))

    # Instantiate metrics object
    metrics = RegressionMetrics(predictionAndLabels)

    print("Mean Squared Error = %s" % metrics.meanSquaredError)
    print("Root Mean Squared Error = %s" % metrics.rootMeanSquaredError)
    print("R-squared = %s" % metrics.r2)
    print("Mean Absolute Error = %s" % metrics.meanAbsoluteError)
    print("Explained variance = %s" % metrics.explainedVariance)

    evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE): %.2f" % rmse)

    return accuracy, rmse

def scikit_metrics(y_test, y_pred):
    y_test = y_test.to_numpy()
    y_pred = np.concatenate(y_pred.to_numpy()).ravel()
    # print("************y_test", y_test)
    # print("************y_pred", y_pred)
    accuracy = np.count_nonzero(y_test == y_pred)/len(y_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print('Accuracy: ', accuracy)
    #print('Recall: ', metrics.recall_score) ERROR, USE NUMPY TO DO THIS.
    #print('Precision: ', metrics.precision_score)
    #print('F1 Measure: ', metrics.f1_score)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', rmse)
    print('R-Squared Error:', metrics.r2_score(y_test, y_pred))
    print('Explained Variance Error:', metrics.explained_variance_score(y_test, y_pred))

    return accuracy, rmse

