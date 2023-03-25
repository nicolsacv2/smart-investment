import tensorflow as tf

class TimeNormalization(tf.keras.layers.Layer):
    def __init__(self, axis = -2):
        super(TimeNormalization, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        self.time_shape = input_shape[self.axis]

    def call(self, inputs):
        std = tf.math.reduce_std(inputs, axis = self.axis)
        std_tensor = tf.where(std == 0., 1., std)
        std_tensor = tf.expand_dims(std_tensor, self.axis)
        std_tensor = tf.repeat(std_tensor, self.time_shape, axis = self.axis)

        mean = tf.math.reduce_mean(inputs, axis = self.axis)
        mean_tensor = tf.expand_dims(mean, self.axis)
        mean_tensor = tf.repeat(mean_tensor, self.time_shape, axis = self.axis)

        output = (inputs - mean_tensor) / std_tensor

        return output, mean, std