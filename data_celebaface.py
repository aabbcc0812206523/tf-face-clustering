from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import tensorflow as tf


def read_record(image_file_queue):

    class CelebaRecord(object):
        pass

  
    result = CelebaRecord()
    height = 64
    width = 64
    depth = 3    
  
    image_bytes = height * width * depth * 1

  
    image_reader = tf.FixedLengthRecordReader(record_bytes=image_bytes)
    _, value = image_reader.read(image_file_queue)
    result.image = tf.decode_raw(value, tf.uint8)
    result.image = tf.reshape(result.image, [height, width, depth])
    result.image = tf.cast(result.image, tf.float32)

    return result

def inputs(data_dir, batch_size):
  
    filename_queue = tf.train.string_input_producer([os.path.join(data_dir, 'celebafaces_%d.bin' % i) for i in range(0, 16)])

    record = read_record(filename_queue)
    record.image = tf.image.random_flip_left_right(record.image)
    # record.image = tf.image.per_image_standardization(record.image)

    return tf.train.batch(
            [record.image],
            batch_size=batch_size,
            num_threads=16,
            capacity=50000)




