from evaluation.EvaluateMetrics import EvaluateMetrics

import os

# Data path
path = {
    'rating': os.path.join('..', 'data', 'ratings.csv'),
    'movie': os.path.join('..', 'data', 'movies.csv')
}

# Configurations
config = {
    'similairty': {'name': 'pearson_baseline', 'user_based': False},
    'test_size': 0.25,
    'random_state': 2,
    'top_n': 10,
    'rating_threshold': 0.4,
    'verbose': True
}

# Initialize class
evaluate = EvaluateMetrics(path=path, config=config)

evaluate.run()
