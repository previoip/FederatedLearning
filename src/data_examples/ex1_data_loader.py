import numpy as np
from src.request_utils import get_request
from src.archive_utils import unload_zip_from_bytes
from pathlib import Path
import pandas as pd


class ExampleDataLoader:

  archive_url = 'https://archive.ics.uci.edu/static/public/10/automobile.zip'
  archive_dir = 'data/ex1'

  data_spec = {
    'symboling':                [-3, -2, -1, 0, 1, 2, 3],
    'normalized-losses':        (65, 256),
    'make':                     ['alfa-romero', 'audi', 'bmw', 'chevrolet', 
                                'dodge', 'honda', 'isuzu', 'jaguar', 'mazda',
                                'mercedes-benz', 'mercury', 'mitsubishi',
                                'nissan', 'peugot', 'plymouth', 'porsche',
                                'renault', 'saab', 'subaru', 'toyota',
                                'volkswagen', 'volvo'],
    'fuel-type':                ['diesel', 'gas'],
    'aspiration':               ['std', 'turbo'],
    'num-of-doors':             ['four', 'two'],
    'body-style':               ['hardtop', 'wagon', 'sedan', 
                                'hatchback', 'convertible'],
    'drive-wheels':             ['4wd', 'fwd', 'rwd'],
    'engine-location':          ['front', 'rear'],
    'wheel-base':               (86.6, 120.9),
    'length':                   (141.1, 208.1),
    'width':                    (60.3,  72.3),
    'height':                   (47.8,  59.8),
    'curb-weight':              (1488,  4066),
    'engine-type':              ['dohc', 'dohcv', 'l', 'ohc', 'ohcf',
                                'ohcv', 'rotor'],
    'num-of-cylinders':         ['eight', 'five', 'four', 'six', 'three',
                                'twelve', 'two'],
    'engine-size':              (61, 326),
    'fuel-system':              ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi',
                                'spdi', 'spfi'],
    'bore':                     (2.54, 3.94),
    'stroke':                   (2.07, 4.17),
    'compression-ratio':        (7,    23),
    'horsepower':               (48,   288),
    'peak-rpm':                 (4150, 6600),
    'city-mpg':                 (13,   49),
    'highway-mpg':              (16,   54),
    'price':                    (5118, 45400)
  }



  data_types_numeric = {
      'symboling'         : np.int8,
      'normalized-losses' : np.float32,
      'wheel-base'        : np.float32,
      'length'            : np.float32,
      'width'             : np.float32,
      'height'            : np.float32,
      'curb-weight'       : np.float32,
      'engine-size'       : np.float32,
      'bore'              : np.float32,
      'stroke'            : np.float32,
      'compression-ratio' : np.float32,
      'horsepower'        : np.float32,
      'peak-rpm'          : np.float32,
      'city-mpg'          : np.float32,
      'highway-mpg'       : np.float32,
      'price'             : np.float32
  }

  label_name = 'symboling'
  losses_name = 'normalized-losses'

  features_categorical = [
    'make',
    # 'fuel-type',
    # 'aspiration',
    # 'num-of-doors',
    # 'body-style',
    # 'drive-wheels',
    # 'engine-location',
    'engine-type',
    # 'num-of-cylinders',
    # 'fuel-system'
  ]

  features_numeric = [    
    "curb-weight",
    "engine-size",
    "horsepower",
    "peak-rpm",
  ]


  def __init__(self):
    self.df = None
    self.data_attrs = self.data_spec.keys()
    self.data_attrs_str = list(filter(lambda x: x not in self.data_types_numeric.keys(), self.data_attrs))

  def download(self, to_folder=None):
    if not to_folder or to_folder is None:
      to_folder = self.archive_dir
    b = get_request(self.archive_url)
    unload_zip_from_bytes(b, to_folder)
    return self

  def load(self):
    data_glob_path = list(Path(self.archive_dir).rglob('*data'))
    data_file_path = list(filter(lambda x: x.is_file(), data_glob_path))[0]
    self.df = pd.read_csv(data_file_path, header=None, names=self.data_attrs, dtype=str, na_values='?')
    return self

  def clean(self):
    self.df = self.df.dropna()
    self.df = self.df.astype(self.data_types_numeric)

    # df['symboling-threshold'] = [1 if i > 0 else 0 for i in df['symboling']]
    # df['symboling-threshold'].astype(np.int32)
    return self