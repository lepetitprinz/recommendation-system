from evaluation.DataPreprocessing import DataPreprocessing
from evaluation.RecommenderMetrics import RecommenderMetrics

from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut


class EvaluateMetrics(object):
	def __init__(self, path: dict, config: dict):
		self.data_prep = DataPreprocessing(path=path)
		self.rec_metrics = RecommenderMetrics()

		# Configuration
		self.verbose = config['verbose']

		# Data configuration
		self.test_size = config['test_size']
		self.random_state = config['random_state']

		# Similarity configuration
		self.sim_options = config['similairty']

		# Recommendation configuration
		self.top_n = config['top_n']
		self.rating_threshold = config['rating_threshold']

	def run(self):
		# 1. Load movie & rating dataset
		rating = self.load_surprise_rating_data()

		# 2. Computing movie popularity rank
		rank = self.get_popularity_rank()

		# 3. Computing item similarities (for measuring diversity)
		sim_estimator, full_train_set = self.compute_similairity(data=rating)

		# 4. Split train & test dataset
		train_set, test_set = self.split_train_test_data(data=rating)

		# 5. Build recommendation model
		rec_estimator = self.build_recommend_model(data=train_set)

		# 6. Evaluate recommendation model
		prediction = self.evaluate_recommend_model(estimator=rec_estimator, data=test_set)

		# 7. Top-N recommendation
		self.recommend_top_n(estimator=rec_estimator, data=rating)

		# 8. All recommendation
		top_n_prediction = self.recommend_all(estimator=rec_estimator, data=full_train_set)

		# 9. Evaluate metrics(User Coverage / Diversity / Novelty)
		self.evaluate_metrics(
			estimator=sim_estimator,
			prediction=top_n_prediction, 
			n_user=full_train_set.n_users
			)

	def load_surprise_rating_data(self):
		data = self.data_prep.load_surprise_dataset()

		return data

	def get_popularity_rank(self):
		rank = self.data_prep.rank_popularity()

		return rank

	def compute_similairity(self, data):
		full_train_set = data.build_full_trainset()
		estimator = KNNBaseline(sim_options=self.sim_options)
		estimator.fit(full_train_set)

		return estimator, full_train_set

	def split_train_test_data(self, data):
		train_data, test_data = train_test_split(
			data, 
			test_size=self.test_size, 
			random_state=self.random_state
			)	

		return train_data, test_data

	def build_recommend_model(self, data):
		# Initialize Singular Vector Decomposition
		estimator = SVD(random_state=self.random_state)
		estimator.fit(data)

		return estimator

	def evaluate_recommend_model(self, estimator, data):
		prediction = estimator.test(data)

		rmse = self.rec_metrics.calc_rmse(prediction)
		mae = self.rec_metrics.calc_mae(prediction)

		if self.verbose:
			print(f"RMSE: {rmse}")
			print(f"MAE: {mae}")

		return prediction

	def recommend_top_n(self, estimator, data):
		# Set aside one rating per user for testing
		loocv = LeaveOneOut(n_splits=1, random_state=self.random_state)

		for train_data, test_data in loocv.split(data):
			# Train model without left-out ratings
			estimator.fit(train_data)

			# Predicts ratings for left-out ratings only
			left_out_prediction = estimator.test(test_data)

			# Build predictions for all ratings not in the training set
			anti_test_set = train_data.build_anti_testset()
			all_prediction = estimator.test(anti_test_set)

			# Compute top-n recommendation for each user
			top_n_prediction = self.rec_metrics.get_top_n(
				prediction=all_prediction,
				n=self.top_n,
				rating_threshold=self.rating_threshold
			)

			# Check how often we recommend a movie the user actually rated
			hit_rate = self.rec_metrics.calc_hit_rate(top_n_prediction, left_out_prediction)

			if self.verbose:
				print(f"Hit Rate: {hit_rate}")

			# Break down the hit rate by rating value
			rating_hit_rate = self.get_rating_hit_rate(
				top_n_prediction=top_n_prediction, 
				left_out_prediction=left_out_prediction
				)

			# Check how often we recommend a movie the user actually like
			cum_hit_rate = self.get_cum_hit_rate(
				top_n_prediction=top_n_prediction, 
				left_out_prediction=left_out_prediction, 
				rating_threshold=self.rating_threshold
				)

			# Check the Average Reciprocal Hit Rank (ARHR)
			arhr = self.get_avg_reciprocal_hit_rank(top_n_prediction, left_out_prediction)

	def get_rating_hit_rate(self, top_n_prediction, left_out_prediction):
		rating_hit_rate = self.rec_metrics.calc_rating_hit_rate(
			top_n_prediction,
			left_out_prediction
		)

		return rating_hit_rate

	def get_cum_hit_rate(self, top_n_prediction, left_out_prediction, rating_threshold):
		cum_hit_rate = self.rec_metrics.calc_cum_hit_rate(
			top_n_prediction,
			left_out_prediction,
			rating_threshold
		)

		return cum_hit_rate

	def get_avg_reciprocal_hit_rank(self, top_n_prediction, left_out_prediction):
		arhr = self.rec_metrics.calc_avg_reciprocal_hit_rank(top_n_prediction, left_out_prediction)

		return arhr


	def recommend_all(self, estimator, data):
		estimator.fit(data)
		anti_testset = data.build_anti_testset()

		# Predict
		all_prediction = estimator.test(anti_testset)
		top_n_prediction = self.rec_metrics.get_top_n(all_prediction, n=self.top_n)


		return top_n_prediction

	def eval_metrics(self, estimator, prediction, n_user, rank):
		# User coverage with a predicted rating of threshold
		user_coverage = self.rec_metrics.calc_user_coverage(
			prediction=prediction, 
			n_user=n_user, 
			threshold=self.rating_threshold
			)

		# Measure the diversity of recommendation
		diversity = self.rec_metrics.calc_diversity(prediction, estimator)


		# Measure novelty (average popularity rank of recommendation)
		novelty = self.rec_metrics.calc_novelty(prediction, rank)
