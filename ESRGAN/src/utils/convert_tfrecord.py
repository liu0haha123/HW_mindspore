from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf


flags.DEFINE_string('hr_dataset_path', './data/DIV2K/DIV2K800_sub',
                    'path to high resolution dataset')
flags.DEFINE_string('lr_dataset_path', './data/DIV2K/DIV2K800_sub_bicLRx4',
                    'path to low resolution dataset')
flags.DEFINE_string('output_path', './data/DIV2K800_sub_bin.tfrecord',
                    'path to ouput tfrecord')
flags.DEFINE_boolean('is_binary', True, 'whether save images as binary files'
                     ' or load them on the fly.')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example_bin(img_name, hr_img_str, lr_img_str):
    # Create a dictionary with features that may be relevant (binary).
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_encoded': _bytes_feature(hr_img_str),
               'image/lr_encoded': _bytes_feature(lr_img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_example(img_name, hr_img_path, lr_img_path):
    # Create a dictionary with features that may be relevant.
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_img_path': _bytes_feature(hr_img_path),
               'image/lr_img_path': _bytes_feature(lr_img_path)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    hr_dataset_path = FLAGS.hr_dataset_path
    lr_dataset_path = FLAGS.lr_dataset_path

    if not os.path.isdir(hr_dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(hr_dataset_path))

    samples = []
    logging.info('Reading data list...')
    for hr_img_path in glob.glob(os.path.join(hr_dataset_path, '*.png')):
        img_name = os.path.basename(hr_img_path).replace('.png', '')
        lr_img_path = os.path.join(lr_dataset_path, img_name + '.png')
        samples.append((img_name, hr_img_path, lr_img_path))
    random.shuffle(samples)

    if os.path.exists(FLAGS.output_path):
        logging.info('{:s} already exists. Exit...'.format(
            FLAGS.output_path))
        exit(1)

    logging.info('Writing {} sample to tfrecord file...'.format(len(samples)))
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_name, hr_img_path, lr_img_path in tqdm.tqdm(samples):
            if FLAGS.is_binary:
                hr_img_str = open(hr_img_path, 'rb').read()
                lr_img_str = open(lr_img_path, 'rb').read()
                tf_example = make_example_bin(img_name=str.encode(img_name),
                                              hr_img_str=hr_img_str,
                                              lr_img_str=lr_img_str)
            else:
                tf_example = make_example(img_name=str.encode(img_name),
                                          hr_img_path=str.encode(hr_img_path),
                                          lr_img_path=str.encode(lr_img_path))
            writer.write(tf_example.SerializeToString())



def _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot):
    def parse_tfrecord(tfrecord):
        if using_bin:
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/hr_encoded': tf.io.FixedLenFeature([], tf.string),
                'image/lr_encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            lr_img = tf.image.decode_png(x['image/lr_encoded'], channels=3)
            hr_img = tf.image.decode_png(x['image/hr_encoded'], channels=3)
        else:
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/hr_img_path': tf.io.FixedLenFeature([], tf.string),
                'image/lr_img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            hr_image_encoded = tf.io.read_file(x['image/hr_img_path'])
            lr_image_encoded = tf.io.read_file(x['image/lr_img_path'])
            lr_img = tf.image.decode_png(lr_image_encoded, channels=3)
            hr_img = tf.image.decode_png(hr_image_encoded, channels=3)

        lr_img, hr_img = _transform_images(
            gt_size, scale, using_flip, using_rot)(lr_img, hr_img)

        return lr_img, hr_img
    return parse_tfrecord


def _transform_images(gt_size, scale, using_flip, using_rot):
    def transform_images(lr_img, hr_img):
        lr_img_shape = tf.shape(lr_img)
        hr_img_shape = tf.shape(hr_img)
        gt_shape = (gt_size, gt_size, tf.shape(hr_img)[-1])
        lr_size = int(gt_size / scale)
        lr_shape = (lr_size, lr_size, tf.shape(lr_img)[-1])

        tf.Assert(
            tf.reduce_all(hr_img_shape >= gt_shape),
            ["Need hr_image.shape >= gt_size, got ", hr_img_shape, gt_shape])
        tf.Assert(
            tf.reduce_all(hr_img_shape[:-1] == lr_img_shape[:-1] * scale),
            ["Need hr_image.shape == lr_image.shape * scale, got ",
             hr_img_shape[:-1], lr_img_shape[:-1] * scale])
        tf.Assert(
            tf.reduce_all(hr_img_shape[-1] == lr_img_shape[-1]),
            ["Need hr_image.shape[-1] == lr_image.shape[-1]], got ",
             hr_img_shape[-1], lr_img_shape[-1]])

        # randomly crop
        limit = lr_img_shape - lr_shape + 1
        offset = tf.random.uniform(tf.shape(lr_img_shape), dtype=tf.int32,
                                   maxval=tf.int32.max) % limit
        lr_img = tf.slice(lr_img, offset, lr_shape)
        hr_img = tf.slice(hr_img, offset * scale, gt_shape)

        # randomly left-right flip
        if using_flip:
            flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
            def flip_func(): return (tf.image.flip_left_right(lr_img),
                                     tf.image.flip_left_right(hr_img))
            lr_img, hr_img = tf.case(
                [(tf.equal(flip_case, 0), flip_func)],
                default=lambda: (lr_img, hr_img))

        # randomly rotation
        if using_rot:
            rot_case = tf.random.uniform([1], 0, 4, dtype=tf.int32)
            def rot90_func(): return (tf.image.rot90(lr_img, k=1),
                                      tf.image.rot90(hr_img, k=1))
            def rot180_func(): return (tf.image.rot90(lr_img, k=2),
                                       tf.image.rot90(hr_img, k=2))
            def rot270_func(): return (tf.image.rot90(lr_img, k=3),
                                       tf.image.rot90(hr_img, k=3))
            lr_img, hr_img = tf.case(
                [(tf.equal(rot_case, 0), rot90_func),
                 (tf.equal(rot_case, 1), rot180_func),
                 (tf.equal(rot_case, 2), rot270_func)],
                default=lambda: (lr_img, hr_img))

        # scale to [0, 1]
        lr_img = lr_img / 255
        hr_img = hr_img / 255

        return lr_img, hr_img
    return transform_images


def load_tfrecord_dataset(tfrecord_name, batch_size, gt_size,
                          scale, using_bin=False, using_flip=False,
                          using_rot=False, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
