import csv
import pandas as pd

from surprise import Dataset
from surprise import Reader


class DataPreprocessing:
	def __init__(self, path):
		self.columns = 'user item rating timestamp'
		self.path = path

		self.movie = None
		self.rating = None

		self.movie_id_to_name = {}
		self.name_to_movie_id = {}

		self.init()

	def init(self):
		self.load()
		self.set_movie_hash_map()

	def load(self):
		self.movie = pd.read_csv(self.path['movie'])
		self.rating = pd.read_csv(self.path['rating'])

	def set_movie_hash_map(self):
		movie = self.movie
		self.movie_id_to_name = movie[['movieId', 'title']].set_index('movieId').to_dict()['title']
		self.name_to_movie_id = movie[['movieId', 'title']].set_index('title').to_dict()['movieId']

	def load_surprise_dataset(self):
		# Rating dataset
		reader = Reader(line_format=self.columns, sep=',', skip_lines=1)
		rating = Dataset.load_from_file(self.path['rating'], reader=reader)

		return rating

	def get_user_rating(self, user):
		user_rating = []
		hit_user = False

		with open(self.path['rating'], newline='') as file:
			rating = csv.reader(file)
			next(rating)
			for row in rating:
				user_id = int(row[0])
				if user == user_id:
					movie_id = int(row[1])
					rate = float(row[2])
					user_rating.append((movie_id, rate))
					hit_user = True
				if hit_user and user != user_id:
					break

			return user_rating

	def rank_popularity(self):
		data = self.rating
		rating = data.groupby(by='movieId')['userId'].count().reset_index()
		rating = rating.sort_values(by=['userId', 'movieId'], ascending=[False, True])
		ranking = {movie_id: i+1 for i, movie_id in enumerate(rating['movieId'])}

		return ranking

	def get_genre(self):
		genre = {}
		genre_map = {}
		max_genre_id = 0
		with open(self.path['movie'], newline='', encoding='ISO-8859-1') as file:
			movie = csv.reader(file)
			next(movie)    # Skip header line
			for row in movie:
				movie_id = int(row[0])
				genre_list = row[2].split('|')
				genre_id_list = []
				for genre in genre_list:
					if genre in genre_map:
						genre_id = genre_map[genre]
					else:
						genre_id = max_genre_id
						genre_map[genre] = genre_id
						max_genre_id += 1
					genre_id_list.append(genre_id)
				genre[movie_id] = genre_id_list

		for movie_id, genre_id_list in genre.items():
			bit_field = [0] * max_genre_id
			for genre_id in genre_id_list:
				bit_field[genre_id] = 1
			genre[movie_id] = bit_field


		return genre

	def get_year(self):
		pass







