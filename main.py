# Imports
import re
import sys
from preprocess import Preprocess

# Constants
parlerDataDirectory = './parler-data.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = 'preprocessed.csv'


def main():
    print("hello world!")

    try:
        preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
        preprocessor.preprocessJson(parlerDataDirectory)
    except:
        sys.exit("Error: Parler data not found.")


def preprocess():
    return 0

if __name__ == "__main__":
    main()