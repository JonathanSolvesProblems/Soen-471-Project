# Imports
import re
import sys
from preprocess import Preprocess

# Constants
parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = 'preprocessed.csv'


def main():
    print("hello world!")

    # exception for testing, move to more appropriate place later.
    try:
        preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
        preprocessor.preprocessJson(parlerDataDirectory)
        # preprocessor.createResultDirectory()
    except:
        sys.exit("Error: Parler data not found or unable to create results.")


def preprocess():
    return 0

if __name__ == "__main__":
    main()