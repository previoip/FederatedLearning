from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

from src.custom_types import TypeEnum

def gen_normalization_layer(tfds, col_name, features=None, axis=None):
  if features is None:
    features = tfds.map(lambda x, _: x[col_name])
  normalizer = tf.keras.layers.Normalization(axis=axis)
  normalizer.adapt(features)
  return normalizer

def gen_multihot_categorical_encoding_layer(tfds, col_name, type_enum: TypeEnum, max_tokens=None, features=None):
  if features is None:
    features = tfds.map(lambda x, _: x[col_name])
  if type_enum == TypeEnum.string:
    indexer = tf.keras.layers.StringLookup(max_tokens=max_tokens)
  elif type_enum == TypeEnum.integer:
    indexer = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
  else:
    raise NameError(f'type_enum is invalid {type_enum}')
  indexer.adapt(features)

  encoder = tf.keras.layers.CategoryEncoding(num_tokens=indexer.vocabulary_size())
  
  return lambda f: encoder(indexer(f))


def gen_string_lookup(tfds, col_name, vocabulary=None, max_tokens=None):
  features = tfds.map(lambda x, _: x[col_name])
  indexer = tf.keras.layers.StringLookup(max_tokens=max_tokens, vocabulary=vocabulary)
  indexer.adapt(features)
  return indexer

def gen_string_embeddings(tfds, col_name, vocabulary=None, max_tokens=None, n_dim=None):
  features = tfds.map(lambda x, _: x[col_name])
  indexer = tf.keras.layers.StringLookup(max_tokens=max_tokens, vocabulary=vocabulary)

  if vocabulary is None: 
    indexer.adapt(features)
    vocabulary = indexer.get_vocabulary()
  
  if n_dim is None:
    n_dim = len(vocabulary)
  
  encoder = tf.keras.layers.Embedding(
    input_dim=len(vocabulary),
    output_dim=int(np.sqrt(n_dim)),
    name=col_name
  )

  return lambda f: encoder(indexer(f)), indexer

  


