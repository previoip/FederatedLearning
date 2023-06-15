from tensorflow.keras import layers
from src.custom_types import TypeEnum

def gen_normalization_layer(tfds, col_name):
  features = tfds.map(lambda x, _: x[col_name])
  normalizer = layers.Normalization(axis=None)
  normalizer.adapt(features)
  return normalizer

def gen_multihot_categorical_encoding_layer(tfds, col_name, type_enum: TypeEnum):
  features = tfds.map(lambda x, _: x[col_name])

  if type_enum == TypeEnum.string:
    indexer = layers.StringLookup()
  elif type_enum == TypeEnum.integer:
    indexer = layers.IntegerLookup()
  else:
    raise NameError(f'type_enum is invalid {type_enum}')
  indexer.adapt(features)

  encoder = layers.CategoryEncoding(num_tokens=indexer.vocabulary_size())
  
  return lambda f: encoder(indexer(f))