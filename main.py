# Imports
import re
import sys
from preprocess import Preprocess
from plot import plotUpvotes

# ~/../../media/jonathan/"Main HDD"/
# './parler_data000000000037.ndjson/'

# Constants
parlerDataDirectory = './parler_data000000000037.ndjson/'
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
    
    upvoteBins = 4
    # exception for testing, move to more appropriate place later.
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)

    
    # preprocessor.reprocessed_data()

    # plotUpvotes(preprocessor.getPreprocessedData(), upvoteBins)
    # plotUpvotes(preprocessor.getProcessedData(), upvoteBins)


def preprocess():
    return 0

if __name__ == "__main__":
    main()