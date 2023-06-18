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

  data_rename_map = {
    'symboling'          : 'symboling',
    'normalized-losses'  : 'normalized_losses',
    'make'               : 'make',
    'fuel-type'          : 'fuel_type',
    'aspiration'         : 'aspiration',
    'num-of-doors'       : 'num_of_doors',
    'body-style'         : 'body_style',
    'drive-wheels'       : 'drive_wheels',
    'engine-location'    : 'engine_location',
    'wheel-base'         : 'wheel_base',
    'length'             : 'length',
    'width'              : 'width',
    'height'             : 'height',
    'curb-weight'        : 'curb_weight',
    'engine-type'        : 'engine_type',
    'num-of-cylinders'   : 'num_of_cylinders',
    'engine-size'        : 'engine_size',
    'fuel-system'        : 'fuel_system',
    'bore'               : 'bore',
    'stroke'             : 'stroke',
    'compression-ratio'  : 'compression_ratio',
    'horsepower'         : 'horsepower',
    'peak-rpm'           : 'peak_rpm',
    'city-mpg'           : 'city_mpg',
    'highway-mpg'        : 'highway_mpg',
    'price'              : 'price',
  }



  data_infer_types = {
    'symboling'          : np.int32,
    'normalized_losses'  : np.float32,
    'make'               : 'string',
    'fuel_type'          : 'string',
    'aspiration'         : 'string',
    'num_of_doors'       : 'string',
    'body_style'         : 'string',
    'drive_wheels'       : 'string',
    'engine_location'    : 'string',
    'wheel_base'         : np.float32,
    'length'             : np.float32,
    'width'              : np.float32,
    'height'             : np.float32,
    'curb_weight'        : np.float32,
    'engine_type'        : 'string',
    'num_of_cylinders'   : 'string',
    'engine_size'        : np.float32,
    'fuel_system'        : 'string',
    'bore'               : np.float32,
    'stroke'             : np.float32,
    'compression_ratio'  : np.float32,
    'horsepower'         : np.float32,
    'peak_rpm'           : np.float32,
    'city_mpg'           : np.float32,
    'highway_mpg'        : np.float32,
    'price'              : np.float32
  }

  feature_label   = 'symboling'
  feature_losses  = 'normalized-losses'

  features_categorical = [
    'make',
    'fuel_type',
    'aspiration',
    'num_of_doors',
    'body_style',
    'drive_wheels',
    'engine_location',
    'engine_type',
    'num_of_cylinders',
    'fuel_system'
  ]

  features_numeric_continuous = [
    'wheel_base',
    'length',
    'width',
    'height',
    'curb_weight',
    'engine_size',
    'bore',
    'stroke',
    'compression_ratio',
    'horsepower',
    'peak_rpm',
    'city_mpg',
    'highway_mpg',
    'price'
  ]


  def __init__(self):
    self.df = None
    self.__is_renamed = False

  def download(self, to_folder=None):
    if not to_folder or to_folder is None:
      to_folder = self.archive_dir
    b = get_request(self.archive_url)
    unload_zip_from_bytes(b, to_folder)
    return self

  def load(self):
    data_glob_path = list(Path(self.archive_dir).rglob('*data'))
    data_file_path = list(filter(lambda x: x.is_file(), data_glob_path))[0]
    self.df = pd.read_csv(data_file_path, header=None, names=self.data_spec.keys(), dtype=str, na_values='?')
    return self

  def clean(self):
    self.df = self.df.dropna()
    self.df = self.df.rename(self.data_rename_map, axis=1)
    self.__is_renamed = True
    self.df = self.df.astype(self.data_infer_types)
    return self

  def drop_unused(self):
    keep = [self.feature_label, self.feature_losses] + self.features_categorical + self.features_numeric_continuous
    others = self.data_spec.keys()
    if self.__is_renamed:
      others = self.data_rename_map.values()
    drops = [i for i in others if i not in keep]

    if drops:
      self.df = self.df.drop(drops, axis=1)

