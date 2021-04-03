# Imports
import re
import sys
from preprocess import Preprocess
from plot import plotUpvotes

# Constants
parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'


def main():
    printArt = ''' _______________
 < Hello, Tristan! >
 ---------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\\
                    ||----w |
                    ||     ||'''
    print(printArt)
    
    upvoteBins = 4
    # exception for testing, move to more appropriate place later.
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    plotUpvotes(preprocessor.getPreprocessedData(), upvoteBins)
    plotUpvotes(preprocessor.getProcessedData(), upvoteBins)


def preprocess():
    return 0

if __name__ == "__main__":
    main()