import tensorflow as tf
import csv
from io import StringIO

def df_to_tfds(df, label_col_name, shuffle=True, batch_size=32):
  """Convert pandas dataframe into tensorflow dataset

  Args:
  """
  c_df = df.copy()
  labels = c_df.pop(label_col_name)
  c_df = {key: value[:,tf.newaxis] for key, value in df.items()}

  tfds = tf.data.Dataset.from_tensor_slices((dict(c_df), labels))

  if shuffle:
    tfds = tfds.shuffle(buffer_size=len(df))

  tfds = tfds.batch(batch_size)
  tfds = tfds.prefetch(batch_size)
  return tfds


def eval_metrics(model, tsdf_test, keras_metrics={}):
  """Evaluate prediction metrics on tensorflow
     test dataset  

  Args:
  """

  ret = {}
  for x_test, y_test in tfds_test:
    y_pred = loaded_model.predict(x_test)
    for metric_name, metric in metrics.items():
      metric.update_state(y_test, y_pred)

  for metric_name, metric in metrics.items():
    ret[metric_name] = metric.result().numpy()
    metric.reset_states()

  return ret


class MetricsLogger(tf.keras.callbacks.Callback):
  def __init__(self, filepath):
    super().__init__()
    print('logging to', filepath)
    self.filepath = filepath
    self.file = None
    self.writer = None

  def on_train_begin(self, logs=None):
    self._records = []

  def on_epoch_end(self, epoch, logs=None):
    row = {'epoch': epoch + 1}
    for k in logs.keys():
      row[k] = logs[k]
    self._records.append(row)

  def on_train_end(self, logs=None):
    fields = self._records[0].keys()
    with open(self.filepath, 'w') as fo:
      self.writer = csv.DictWriter(fo, fields, extrasaction='raise')
      self.writer.writeheader()
      for row in self._records:
        self.writer.writerow(row)
      fo.flush()

  def _get_records(self):
      return self._records