import tensorflow as tf
import os.path

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        context={
            'video_id': tf.FixedLenFeature([], tf.string),
            'start_time_seconds': tf.FixedLenFeature([], tf.float32),
            'end_time_seconds': tf.FixedLenFeature([], tf.float32),
            'labels': tf.FixedLenFeature([], tf.int64)
        }
    )

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(context['labels'], tf.int32)

    return label

def main():
