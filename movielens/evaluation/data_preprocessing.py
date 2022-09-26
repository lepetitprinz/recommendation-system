import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict


class MovieLens:


	def __init__(self):
		self.columns = 'user item rating timestamp'
		self.path = {
			'rating': os.path.join('..', 'data', 'ratings.csv')
			'movie': os.path.join('..', 'data', 'movies.csv')
		}
		self.movie_id_to_name = {}
		self.name_to_movie_id = {}


	def load_dataset(self):
		# Rating dataset
		reader = Reader(line_format=self.columns, sep=',', skip_lines=1)
		rating = Dataset.load_from_file(self.path['rating'], reader=reader)

		# Make movie id <-> name hash map
		with open(self.path['movie'], newline='', encoding='ISO-8859-1') as file:
			movie = csv.reader(file)
			next(movie)
			for row in movie:
				movie_id = int(row[0])
				movie_name = row[1]
				self.movie_id_to_name[movie_id] = movie_name
				self.name_to_movie_id[movie_name] = movie_id

		return rating

	def get_user_rating(self, user):
		user_rating = []
		hit_user = False

		with open(self.path['rating'], newline='') as file:
			rating = csv.reader(file):
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

	def get_popularity_rank(self):
		rating = {}
		ranking = {}

		with open(self.path['rating'], newline='') as file:
			rating = csv.reader(file)
			next(rating)
			for row in rating:
				movie_id = int(row[1])
				if movie_id in ratings:
					rating[movie_id] += 1
				else:
					rating[movie_id] = 1

		rank = 1
		rating = sorted(rating.items(), key=lambda x: x[1], reverse=True)
		for movie_id, rating_cnt in rating:
			ranking[movie_id] = rank
			rank += 1

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
		







