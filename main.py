# Imports
import re
import sys
from preprocess import Preprocess

# Constants
parlerDataDirectory = './part-00000-fe953ba9-c2ae-401a-bf98-58cbafc0ddcc-c000.csv/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'


def main():
    print("hello world! we ride at dawn")

    # exception for testing, move to more appropriate place later.
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)
    # preprocessor.createResultDirectory()


def preprocess():
    return 0

if __name__ == "__main__":
    main()