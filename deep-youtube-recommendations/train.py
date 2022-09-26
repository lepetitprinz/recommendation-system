import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from core.dataset import Dataset
import core.config as cfg
from model.network import DeepRecommendations
from model.candidate_generation import CandidateGeneration


class Train(object):
	def __init__(self):
		self.trainset = Dataset('train').data
		self.testset = Dataset('test').data
		self.model = CandidateGeneration().build_nework()
		# self.model = DeepRecommendations(self.trainset)

	def train(self):

		# self.model.fit()
		history = self.model.fit([pad_sequences(self.trainset['movie_id']),
								  pad_sequences(self.trainset['watch_hist_time'], dtype=float),
								  pad_sequences(self.trainset['search_hist'], dtype=float) + 1e-10,
								  pad_sequences(self.trainset['example_age'], dtype=float),],
								  self.trainset['predict_labels'].values, steps_per_epoch=1, epochs=50)


if __name__ == '__main__': 
	Train().train()
