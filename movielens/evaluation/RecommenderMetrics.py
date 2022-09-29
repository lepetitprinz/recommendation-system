from surprise import accuracy


class RecommenderMetrics(object):
	@staticmethod
	def calc_mae(prediction):
		return accuracy.mae(prediction, verbose=False)

	@staticmethod
	def calc_rmse(prediction):
		return accuracy.rmse(prediction, verbose=False)

	@staticmethod
	def get_top_n(prediction, n, rating_threshold):
		top_n = {}
		for user_id, movie_id, actual_rating, estimate_rating, _ in prediction:
			if estimate_rating >= rating_threshold:
				user_id = int(user_id)
				if user_id in top_n:
					top_n[user_id].append((int(movie_id), estimate_rating))
				else:
					top_n[user_id] = [(int(movie_id), estimate_rating)]

		for user_id, rating in top_n.items():
			rating = sorted(rating, key=lambda x: x[1], reverse=True)
			top_n[user_id] = rating[:n]

		return top_n

	@staticmethod
	def calc_hit_rate(top_n_prediction, left_out_prediction):
		hit_cnt = 0
		total_cnt = 0
		for left_out in left_out_prediction:
			user_id = int(left_out[0])
			left_out_movie_id = int(left_out[1])

			for movie_id, predict_rating in top_n_prediction[user_id]:
				if left_out_movie_id == int(movie_id):
					hit_cnt += 1
					break

			total_cnt += 1


		return hit_cnt / total_cnt

	@staticmethod
	def calc_cum_hit_rate(top_n_prediction, left_out_prediction, rating_cut_off):
		hit_count = 0
		total_cnt = 0

		for user_id, left_out_movie_id, actual_rating, estimate_rating, _ in left_out_prediction:
			if actual_rating >= rating_cut_off:
				for movie_id, predict_rating in top_n_prediction[int(user_id)]:
					if int(left_out_movie_id) == movie_id:
						hit_count += 1
						break

				total_cnt += 1


		return hit_cnt / total_cnt

	@staticmethod
	def calc_rating_hit_rate():
		pass
