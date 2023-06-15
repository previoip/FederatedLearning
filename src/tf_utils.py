import tensorflow as tf

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