import tensorflow as tf

def transform_list(timeseries: list, input_timesteps: int, output_timesteps: int) -> tf.Tensor:
    len_timeseries = len(timeseries)
    input_timeseries = []
    output_timeseries = []
    for index in range(len_timeseries - (input_timesteps + output_timesteps - 1)):
        input_timeseries_item = timeseries[index:(index + input_timesteps)]
        output_timeseries_item = timeseries[(index + input_timesteps):(index + input_timesteps + output_timesteps)]
        input_timeseries.append(input_timeseries_item)
        output_timeseries.append(output_timeseries_item)
    return tf.convert_to_tensor(input_timeseries, dtype=tf.float32), tf.convert_to_tensor(output_timeseries, dtype=tf.float32)

def input_expand_dims(input_tensor, output_tensor):
    return tf.expand_dims(input_tensor, axis=-1), output_tensor


def convert_dataframe_to_tfdataset(dataframe, periods):
    dataset = tf.data.Dataset.from_tensor_slices(transform_list(dataframe['#Passengers'].tolist(), periods, 1))
    dataset = dataset.map(input_expand_dims)
    return dataset

def long_term_predict(model, initial_data, timesteps=40):
    initial_shape = initial_data.shape
    new_data = initial_data
    step = 0
    while step < timesteps:
        input_data = new_data[:, -initial_shape[-2]:, :]
        simple_pred = model.predict(input_data, verbose=0)
        new_data = tf.concat([new_data, tf.cast(tf.expand_dims(simple_pred, axis=0), tf.float32)], axis=-2)
        step += 1
    return new_data[:, -timesteps:, :]