import tensorflow as tf
from src.tf_layer_constructors import (
  gen_multihot_categorical_encoding_layer,
  gen_normalization_layer
)
from src.tf_utils import MetricsLogger
from src.utils import dict_to_flat_list
from src.custom_types import TypeEnum
from os.path import join as joinpath

def generate_input_layers(tfds_train, data_loader):
  all_inputs = {}
  all_encoded_features = {}


  all_inputs['normalization'] = {}
  all_encoded_features['normalization'] = {}

  for col_name in data_loader.features_numeric_continuous:

    input_numeric = tf.keras.Input(shape=(1,), name=col_name, dtype='float32')
    normalization_layer = gen_normalization_layer(tfds_train, col_name)
    encoded_normalized_input = normalization_layer(input_numeric)

    all_inputs['normalization'][col_name] = input_numeric
    all_encoded_features['normalization'][col_name] = encoded_normalized_input


  all_inputs['categorical'] = {}
  all_encoded_features['categorical'] = {}

  for col_name in data_loader.features_categorical:
    input_categorical = tf.keras.Input(shape=(1,), name=col_name, dtype='string')
    categorical_encoder = gen_multihot_categorical_encoding_layer(tfds_train, col_name, TypeEnum.string)
    encoded_categorical_input = categorical_encoder(input_categorical)

    all_inputs['categorical'][col_name] = input_categorical
    all_encoded_features['categorical'][col_name] = encoded_categorical_input

  return all_inputs, all_encoded_features


def build_model(tfds, data_loader, name='model'):

  input_layers, encoded_layers = generate_input_layers(tfds, data_loader)
  input_layers = dict_to_flat_list(input_layers, ls=[])
  encoded_layers = dict_to_flat_list(encoded_layers, ls=[])
  feature_layers = tf.keras.layers.concatenate(encoded_layers)


  x = tf.keras.layers.Dense(64, activation="relu")(feature_layers)
  x = tf.keras.layers.Dropout(0.5)(x)
  output = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(input_layers, output, name=name)
  return model


def compile_model(
    model,
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', 'mse']
  ):
  model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
  return model


def train_and_log_metrics(
    model,
    tfds_train,
    tfds_val,
    epoch,
    csv_metrics=''
  ):
  if model.name != 'model':
    filename = f'{model.name}_metrics.csv'

  logger = MetricsLogger(filename=filename, _metrics=['accuracy', 'mse'])
  history = model.fit(tfds_train, epochs=epoch, validation_data=tfds_val, callbacks=[logger], verbose=0)
  return model, logger, history

def eval_example_data(
    tfds_train,
    tfds_val,
    data_loader,
    epoch=100,
    model_name='model',
    csv_metrics_path='metrics',
  ):
  model = build_model(tfds_train, data_loader, name=model_name)
  compile_model(model)
  _model, logger, history = train_and_log_metrics(model, tfds_train, tfds_val, epoch, csv_metrics=csv_metrics_path)
  return model, logger, history
