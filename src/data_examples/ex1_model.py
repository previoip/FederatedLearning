import tensorflow as tf

class AutoModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.dense = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dropout = tf.keras.layers.Dropout(.5)
    self._output = tf.keras.layers.Dense(1)

  def call(self, inputs, training=False):
    x = self.dense(inputs)
    x = self.dropout(x)
    return self._output(x)