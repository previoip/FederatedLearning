import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow_datasets as tf_dataset

class ExampleDataLoader:

  # MovieLens data loader
  # permalink: https://grouplens.org/datasets/movielens/latest/

  archive_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

  archive_dir = 'data/ex2'

  # for simplicity, loader uses tfgs loader from tensorflow-dataset package.
  # does not need to be nightly build. Link is still available as attr above
  # for documentation

  def __init__(self):
    self.df_ratings = None
    self.df_movie = None
    self._tfds_prefetch_ratings = None
    self._tfds_prefetch_movies = None

  def download(self, to_folder=None):

    self._tfds_prefetch_ratings = tf_dataset.load(
      'movielens/100k-ratings',
      split='train',
      shuffle_files=True,
      data_dir=self.archive_dir
    )

    self._tfds_prefetch_movies = tf_dataset.load(
      'movielens/100k-movies',
      split='train',shuffle_files=True,
      data_dir=self.archive_dir
    )

    return self

  def load(self):
    self.df_ratings = self._tfds_prefetch_ratings
    self.df_movie   = self._tfds_prefetch_movies
