# Imports
import re
import sys
from preprocess import Preprocess

# Constants
parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'


def main():
    print("hello world!")

    # exception for testing, move to more appropriate place later.
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    # preprocessor.createResultDirectory()
    df = preprocessor.getProcessedData()
    df = df.drop('body')
    df.show()

def preprocess():
    return 0

if __name__ == "__main__":
    main()