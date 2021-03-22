# Imports
import re
from preprocess import Preprocess

# Constants
parlerDataDirectory = './parler-data/'
outputFileDirectory = './preprocessed/'
outputJson = 'preprocessed.csv'


def main():
    print("hello world!")
    preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
    preprocessor.preprocessJson(parlerDataDirectory)

def preprocess():
    return 0;

if __name__ == "__main__":
    main()