# Imports
import re
import sys
from preprocess import Preprocess
from plot import plotUpvotes
from optimizedModels import *
from models import *
# ~/../../media/jonathan/"Main HDD"/
# parler_data000000000037.ndjson
# Constants
parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parler-data/'


def main():
    printArt = \
'''
                      o         _         _            _          
           _o        /\_      _ \\o      (_)\__/o     (_)
         _< \_      _>(_)    (_)/<_        _| \      _|/' \/
        (_)>(_)    (_)           (_)      (_)       (_)'  _\o_
    -----------------------------------------------------------------
'''
    print(printArt)
    
    upvoteBins = 20
    # exception for testing, move to more appropriate place later.
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)

    
    # preprocessor.reprocessed_data()

    # plotUpvotes(preprocessor.getPreprocessedData(), upvoteBins)
    # plotUpvotes(preprocessor.getProcessedData(), upvoteBins)

    predictions = random_forest_spark(parlerDataDirectory)
    # random_forest_scikit(parlerDataDirectory)
    # linear_regression_spark(parlerDataDirectory)
    # train_x, train_y, test_x, test_y = prep_data_scikit(parlerDataDirectory)
    # model, test_y_pred, test_comparison = linear_regression_scikit(parlerDataDirectory)
    # scikit_metrics(test_y, test_y_pred)
    # random_forest_spark_metrics(predictions)
    # gridsearch_rf_scikit(parlerDataDirectory)
    # gridsearch_rf_spark(parlerDataDirectory)


def preprocess():
    return 0

if __name__ == "__main__":
    main()