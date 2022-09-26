from DataPreprocesing import DataPreprocesing
from RecommenderMetrics import RecommenderMetrics

from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise

config = {
	'similairty' : {'name': 'pearson_baseline', 'user_based': False}
}


class Evaluate(object):
	verbose = True
	def __init__(self, path, config):
		self.data_prep = DataPreprocesing(path=path)
		self.rec_metrics = RecommenderMetrics()
		self.sim_options = config['similairty']
		self.test_size = 0.25
		self.random_state = 2
		self.top_n = 10
		self.rating_cut_off = 4.0
		self.rating_threshold = 4.0

	def run(data):
		# 1. Load movie & rating dataset
		data = self.load_data()

		# 2. Computing movie popularity rank
		rank = self.get_popularity_rank()

		# 3. Computing item similarities (for measuring diversity)
		sim_estimator, all_data = self.compute_similairity(data=data)

		# 4. Split train & test dataset
		train_data, test_data = self.split_train_test_data(data=data)

		# 5. Build recommendation model
		rec_estimator = self.build_recommend_model(data=train_data):

		# 6. Evaluate recommendation model
		prediction = self.evaluate_recommend_model(estimator=rec_estimator, data=test_data)

		# 7. Top-N recommendation
		self.recommend_top_n(estimator=rec_estimator, data=data)

		# 8. All recommendation
		top_n_prediction = self.recommend_all(estimator=rec_estimator, data=all_data)

		# 9. Evaluate metrics(User Coverage / Diversity / Novelty)
		self.evaluate_metrics(
			estimator=sim_estimator,
			prediction=top_n_prediction, 
			n_user=all_data.n_users
			)

	def load_data(self):
		data = self.data_prep.load_data()

		return dataset

	def get_popularity_rank():
		rank = self.data_prep.calc_popularity_rank()

		return rank

	def compute_similairity(data):
		all_data = data.build_all_train_data()
		estimator = KNNBaseline(sim_options=self.sim_options)
		estimator.fit(all_data)

		return estimator, all_data

	def split_train_test_data(data):
		train_data, test_data = train_test_split(
			data, 
			test_size=self.test_size, 
			random_state=self.random_state
			)	

		return train_data, test_data


	def build_recommend_model(data):

		estimator = SVD(random_state=self.random_state)
		estimator.fit(data)

		return estimator

	def evaluate_recommend_model(estimator, data):
		prediction = estimator.test(data)

		rmse = self.rec_metrics.calc_rmse(prediction)
		mae = self.rec_metrics.calc_mae(prediction)

		if self.verbose
			print(f"RMSE: {rmse}")
			print(f"MAE: {mae}")

		return prediction

	def recommend_top_n(estimator, data):
		# Set aside one rating per user for testing
		loocv = LeaveOneOut(n_split=1, random_state=self.random_state)

		for train_data, test_data in loocv.split(data):
			# Train model without left-out ratings
			estimator.fit(train_data)

			# Predicts ratings for left-out ratings only
			left_out_prediction = estimator.test(test_data)

			# Build predictions for all ratings not in the training set
			big_test_data = train_data.build_anti_testset()
			all_prediction = estimator.test(big_test_data)

			# Compute top-n recommendation for each user
			top_n_prediction = self.rec_metrics.get_top_n(all_prediction, n=self.top_n)

			# Check how often we recommend a movie the user actually rated
			hit_rate = self.rec_metrics.calc_hit_rate(top_n_prediction, left_out_prediction)

			if verbose:
				print(f"Hit Rate: {hit_rate}")

			# Break down the hit rate by rating value
			rating_hit_rate = self.get_rating_hit_rate(
				top_n_prediction=top_n_prediction, 
				left_out_prediction=left_out_prediction
				)

			# Check how often we recommend a movie the user actually like
			cum_hit_rate = self.get_cum_hit_rate(top_n_prediction, left_out_prediction, ratingCutoff=self.rating_cut_off)

			# Check the Average Reciprocal Hit Rank (ARHR)
			arhr = self.get_avg_reciprocal_hit_rank(top_n_prediction, left_out_prediction)


	def get_rating_hit_rate(top_n_prediction, left_out_prediction):
		rating_hit_rate = self.rec_metrics.calc_rating_hit_rate(top_n_prediction, left_out_prediction)

		return rating_hit_rate

	def get_cum_hit_rate(top_n_prediction, left_out_prediction, ratingCutoff):
		cum_hit_rate = self.rec_metrics.calc_cum_hit_rate(top_n_prediction, left_out_prediction, ratingCutoff)

		return cum_hit_rate

	def get_avg_reciprocal_hit_rank(top_n_prediction, left_out_prediction):
		arhr = self.rec_metrics.calc_avg_reciprocal_hit_rank(top_n_prediction, left_out_prediction)

		return arhr


	def recommend_all(estimator, data):
		estimator.fit(data)
		big_test_data = data.build_anti_testset()

		# Predict
		all_prediction = estimator.test(big_test_data)
		top_n_prediction = self.rec_metrics.get_top_n(all_prediction, n=self.top_n)


		return top_n_prediction

	def eval_metrics(estimator, prediction, n_user, rank):
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
