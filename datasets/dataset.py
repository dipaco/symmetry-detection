import os

import numpy as np
import tensorflow as tf

from glob import glob


class Dataset(object):
    def __init__(self, basedir):
        self.basedir = os.path.join(basedir, '')
        self.label_map = {}

    def get_trainfiles(self, ext='off'):
        self.trainfiles = glob(self.basedir + '/*/train/*.' + ext, recursive=True)

        
    def get_testfiles(self, ext='off'):
        self.testfiles = glob(self.basedir + '/*/test/*.' + ext, recursive=True)

    def build_tfrecord(self, tfrecordname, files):
        pass

    def to_tfrecord(self):
        pass

    def from_tfrecord(self):
        pass

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def shuffle(self):
        n = len(self.trainfiles)
        perm = np.random.permutation(n)
        return [ self.trainfiles[idx] for idx in perm ]

    def shuffle(self, data):
        n = len(data)
        perm = np.random.permutation(n)
        return [ data[idx] for idx in perm ]

    @staticmethod
    def get_dataset(FLAGS, train_test, map_function):
        #num_shards = int(FLAGS.train_size / FLAGS.num_samples)

        tf_record_list = glob(os.path.join(FLAGS.data_dir, '{}1024_*.tfrecord'.format(train_test)))
        print('Tensorflow records for mode ({}):\n{}'.format(train_test, tf_record_list))
        dataset = tf.data.TFRecordDataset(
            tf_record_list,
            num_parallel_reads=os.cpu_count()
        )

        dataset = dataset.map(map_func=map_function)
        #dataset = dataset.filter(
        #    lambda w, x, y, z, u, v, f: tf.less_equal(tf.shape(w)[0], tf.cast(FLAGS.max_mesh_size, tf.int32)))
        dataset = dataset.map(map_func=lambda w, x, y, z, u, v, f: [w,
                                                                    tf.transpose(x, perm=[0, 2, 1]),
                                                                    tf.expand_dims(y, -1),
                                                                    z,
                                                                    u,
                                                                    v,
                                                                    [f]])
        # dataset = dataset.batch(1)
        dataset = dataset.padded_batch(1, ([None, 3], [None, 3, 3], [1024, 3, 1], [3, 3, 3], [3], [], [1]))
        # dataset = dataset.batch(1)
        # dataset = dataset.shard(num_shards, 0)
        # dataset = dataset.apply(tf.data.experimental.filter_for_shard(num_shards, 0))

        return dataset
