import numpy as np
from src.request_utils import get_request
from src.archive_utils import unload_zip_from_io
from pathlib import Path
import pandas as pd


def _date_parser(datestr):
  __month_shortname_index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  s = datestr.split('-')
  if len(s) != 3:
    return pd.NaT
  s[0] = '{:02}'.format(int(s[0]))
  s[1] = '{:02}'.format(__month_shortname_index.index(s[1]) + 1)
  s[2], s[0] = s[0], s[2]
  return np.datetime64('-'.join(s))

class ExampleDataLoader:
  # MovieLens data loader
  # permalink: https://grouplens.org/datasets/movielens/latest/

  archive_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
  archive_dir = 'data/ex2'

  data_dir = {
    'users': {
      'filename': 'u.user',
      'delimiter': '|',
      'encoding': 'latin-1',
      'headers': ['user_id', 'age', 'sex', 'occupation', 'zip_code'],
      'types': ['uint64', 'int32', 'string', 'string', 'string']
    },
    'ratings': {
      'filename': 'u.data',
      'delimiter': '\t',
      'encoding': 'latin-1',
      'headers': ['user_id', 'movie_id', 'rating', 'unix_timestamp'],
      'types': ['uint64', 'uint64', 'float64', 'uint64']
    },
    'movies': {
      'filename': 'u.item',
      'delimiter': '|',
      'encoding': 'latin-1',
      'headers': ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
      'genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
      ],
      'types': ['uint64', 'string', 'string', 'string', 'string',
      'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8',
      'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8',
      'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8'
      ]
    },
  }

  data_feature_cols = {
    'users': ['age', 'sex', 'occupation', 'zip_code'],
    'ratings': ['rating', 'unix_timestamp'],
    'movies' : ['title', 'release_date', 'video_release_date']
  }

  data_movies_genres = [
    'genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
  ]

  data_converters = {
    'users': {
      'user_id': lambda x: str(int(x)-1)
    },
    'ratings': {
      'unix_timestamp': lambda x: np.datetime64('1970-01-01') + np.timedelta64(x, 's'),
      'user_id': lambda x: str(int(x)-1),
      'movie_id': lambda x: str(int(x)-1),
    },
    'movies': {
      'movie_id': lambda x: str(int(x)-1),
      'release_date': _date_parser
    }
  }


  def __init__(self):
    self.df = None
    self.df_users = None
    self.df_ratings = None
    self.df_movies = None

  def download(self, to_folder=None):
    if not to_folder or to_folder is None:
      to_folder = self.archive_dir
    else:
      self.archive_dir = to_folder
    b = get_request(self.archive_url)
    unload_zip_from_io(b, to_folder)
    return self

  def load(self):
    for tb_name, tb_info in self.data_dir.items():
      data_glob_path = list(Path(self.archive_dir).rglob(tb_info['filename']))
      data_file_path = next(filter(lambda x: x.is_file(), data_glob_path))
      df = pd.read_csv(
        data_file_path,
        names=tb_info['headers'],
        sep=tb_info['delimiter'],
        encoding=tb_info['encoding'],
        dtype=dict(zip(tb_info['headers'], tb_info['types'])),
        converters=self.data_converters.get(tb_name)
      )
      setattr(self, f'df_{tb_name}', df)
    return self

  def clean(self):
    self.df = self.df_ratings.merge(self.df_movies, on='movie_id')
    self.df = self.df.merge(self.df_users, on='user_id')