from tensorflow.keras import layers
import tensorflow as tf

from src.custom_types import TypeEnum

from enum import Enum, auto

class PreprocessingLayerEnum(Enum):
  text_vectorization = auto()
  normalization      = auto()
  discretization     = auto()
  category_encoding  = auto()
  hashing            = auto()
  string_lookup      = auto()
  integer_lookup     = auto()


class PreprocessingLayerConstructor:

  @classmethod
  def infer_method(preprocessing_layer_enum: PreprocessingLayerEnum) -> callable:
    if preprocessing_layer_enum not in PreprocessingLayerEnum:
      raise ValueError(f'enum is not in PreprocessingLayerEnum {preprocessing_layer_enum}')
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.text_vectorization:
      raise NotImplementedError('preprocessing layer is not yet implemented')
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.normalization:
      return PreprocessingLayerConstructor.gen_normalization_layer
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.discretization:
      raise NotImplementedError('preprocessing layer is not yet implemented')
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.category_encoding:
      return PreprocessingLayerConstructor.gen_multihot_categorical_encoding_layer
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.hashing:
      raise NotImplementedError('preprocessing layer is not yet implemented')
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.string_lookup:
      return tf.keras.layers.StringLookup
    
    elif preprocessing_layer_enum == PreprocessingLayerEnum.integer_lookup:
      return tf.keras.layers.IntegerLookup

    else:
      raise NameError('undefined error')


  @classmethod
  def gen_normalization_layer(tfds, col_name, features=None, axis=None):
    if features is None:
      features = tfds.map(lambda x, _: x[col_name])
    normalizer = tf.keras.layers.Normalization(axis=axis)
    normalizer.adapt(features)
    return normalizer

  @classmethod
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


    @classmethod
    def gen_discretization_encoding_layer(tfds, col_name, num_bins, epsilon=.01):
      if features is None:
        features = tfds.map(lambda x, _: x[col_name])
      # max_v = features.reduce(
      #   tf.cast
      # )
      discretize_encoder = tf.keras.layers.Discretization(num_bins=num_bins, epsilon=epsilon)

      return discretize_encoder


  
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