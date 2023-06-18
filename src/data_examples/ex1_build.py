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
  """Generate input layers based on customized data_loader attributes.

  Args:
      tfds_train: TensorFlow Dataset for training
      data_loader: data loader class instance 
  """

  # layers are stored in dictionary for verbosity, traceability, 
  # and debugging purposes

  all_inputs = {}
  all_encoded_features = {}


  # Normalization step for numeric features, please note alternative
  # to normalization might be discretization, but it has not yet been
  # implemented in this case

  all_inputs['normalization'] = {}
  all_encoded_features['normalization'] = {}

  for col_name in data_loader.features_numeric_continuous:

    input_numeric = tf.keras.Input(shape=(1,), name=col_name, dtype='float32')
    normalization_layer = gen_normalization_layer(tfds_train, col_name)
    encoded_normalized_input = normalization_layer(input_numeric)

    all_inputs['normalization'][col_name] = input_numeric
    all_encoded_features['normalization'][col_name] = encoded_normalized_input


  # for categorical features, since dataset is composed of
  # features with string dtype, thus input layer uses 
  # StringLookup -> CategoryEncoder Sequential layers

  all_inputs['categorical'] = {}
  all_encoded_features['categorical'] = {}

  for col_name in data_loader.features_categorical:
    input_categorical = tf.keras.Input(shape=(1,), name=col_name, dtype='string')
    categorical_encoder = gen_multihot_categorical_encoding_layer(tfds_train, col_name, TypeEnum.string)
    encoded_categorical_input = categorical_encoder(input_categorical)

    all_inputs['categorical'][col_name] = input_categorical
    all_encoded_features['categorical'][col_name] = encoded_categorical_input

  return all_inputs, all_encoded_features

def build_input_layers(tfds, data_loader):
  """Generator for input layers.

  Input layers constructed from generate_input_layers() yields 
  python dictionary. This step flattens the dictionary and
  concatenate encoded layers into one input representation 
  into the model

  Args:
      tfds_train: TensorFlow Dataset for training
      data_loader: data loader class instance 
  """
  input_layers, encoded_layers = generate_input_layers(tfds, data_loader)
  input_layers = dict_to_flat_list(input_layers, ls=[])
  encoded_layers = dict_to_flat_list(encoded_layers, ls=[])
  feature_layers = tf.keras.layers.concatenate(encoded_layers)

  return input_layers, feature_layers

def build_model(tfds, data_loader, name='model', activation='sigmoid'):
  """Generator for the model.

  Customize this generator for other types of algorithm
  that fits the project requirements

  Args:
      tfds_train: TensorFlow Dataset for training
      data_loader: data loader class instance
      name[optional]: the name of the model for logger 
        or other callbacks
      activation[optional]: the activation function to 
      pass through. Defaults to `sigmoid` for logistic
      regression
  """
  input_layers, feature_layers = build_input_layers(tfds, data_loader)

  x = tf.keras.layers.Dense(32, activation=activation)(feature_layers)
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
    csv_metrics_filepath = 'metrics'
  ):

  filename = f'{model.name}_training_metrics.csv'
  csv_metrics_filepath = joinpath(csv_metrics_filepath, filename)

  logger = MetricsLogger(filepath=csv_metrics_filepath)
  history = model.fit(tfds_train, epochs=epoch, validation_data=tfds_val, callbacks=[logger], verbose=0)
  return model, logger, history

def build_and_evaluate(
    tfds_train,
    tfds_val,
    data_loader,
    epoch=100,
    model_name='model',
    csv_metrics_filepath ='metrics',
  ):

  """Composite function, or a wrapper, for all of the above.

  A Wrapper for ease of modelling and training of a given dataset
  into the generator made previously.
  """

  model_metrics = ['accuracy', 'mse', tf.keras.metrics.BinaryAccuracy()]

  model = build_model(
    tfds_train,
    data_loader,
    name=model_name
  )

  compile_model(model, metrics=model_metrics)

  _model, logger, history = train_and_log_metrics(
    model,
    tfds_train,
    tfds_val,
    epoch,
    csv_metrics_filepath=csv_metrics_filepath
  )

  return model, logger, history
