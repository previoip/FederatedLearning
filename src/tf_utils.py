import tensorflow as tf
import csv
from io import StringIO

def df_to_tfds(df, label_col_name, shuffle=True, batch_size=32):
  c_df = df.copy()
  labels = c_df.pop(label_col_name)
  c_df = {key: value[:,tf.newaxis] for key, value in df.items()}

  tfds = tf.data.Dataset.from_tensor_slices((dict(c_df), labels))

  if shuffle:
    tfds = tfds.shuffle(buffer_size=len(df))

  tfds = tfds.batch(batch_size)
  tfds = tfds.prefetch(batch_size)
  return tfds


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename, _metrics=['accuracy']):
        super().__init__()
        self.filename = filename
        self.file = None
        self.writer = None
        self._metrics=_metrics
        self._records = []
        print('logging to', filename)

    def on_train_begin(self, logs=None):
        self.file = open(self.filename, 'w')
        self.writer = csv.DictWriter(self.file, ['epoch', 'loss'] + self._metrics)
        self.writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        row = {'epoch': epoch + 1}

        for k in self._metrics + ['loss']:
            row[k] = logs[k]

        self._records.append(row)
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, logs=None):
        self.file.close()

    def _get_records(self):
        return self._records