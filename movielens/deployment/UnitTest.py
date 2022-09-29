from evaluation.DataPreprocessing import DataPreprocessing

import os
from time import time

# Data path
path = {
    'rating': os.path.join('..', 'data', 'ratings.csv'),
    'movie': os.path.join('..', 'data', 'movies.csv')
}

data_prep = DataPreprocessing(path=path)

data = data_prep.load_surprise_dataset()

rank = data_prep.rank_popularity()
