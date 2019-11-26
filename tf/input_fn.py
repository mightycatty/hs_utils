"""
用于tf keras训练的tf dataset pineline，需要实现生成tf record
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def input_fn_from_tfrecord(tfrecord_filenames_list, tf_example_parser_fn, batch_size=32, map_cores=4, augmentation=False,
                           aug_fn=None, debug=False, shuffle_flag=False, shuffle_buffer_size=1000, **kwargs):
    """
    simple code pipeline for building input_fn consuming tf records
    :param tfrecord_filenames_list: dir to a tf record of list of which
    :param tf_example_parser_fn: fn for parse tf sample from tf record
    :param batch_size: batch size
    :param map_cores: num of corse for parallism
    :param augmentation: augmentation flag
    :param aug_fn: augmentation fn
    :param debug: debug flag
    :param shuffle_buffer_size:
    :param kwargs:
    :return:
    """
    if type(tfrecord_filenames_list) is str:
        tfrecord_filenames_list = [tfrecord_filenames_list]
    # read data from tf record
    num_parallel_reads = len(tfrecord_filenames_list)
    dataset = tf.data.TFRecordDataset(tfrecord_filenames_list, num_parallel_reads=num_parallel_reads,
                                      buffer_size=None)
    dataset = dataset.prefetch(buffer_size=batch_size * 2)
    # avoid shuffle to stabilize val metrics
    if shuffle_flag:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()
    # Parse the record into tensors.
    dataset = dataset.map(tf_example_parser_fn, num_parallel_calls=map_cores)
    # augmentation
    if augmentation:
        assert aug_fn is not None, 'augmentation function must be provided if augmentation flag is true'
        dataset = dataset.map(aug_fn, num_parallel_calls=map_cores)
    dataset = dataset.batch(batch_size)
    if debug:
        iterator = dataset.make_one_shot_iterator()
        dataset = iterator.get_next()
    return dataset


def test_single_frame():
    """
    检查单帧的tfrecord是否正常
    :return:
    """
    tf_record_path = [r'F:\heshuai\proj\matting_tf\dataset_utils\generators\sp_ful.record',
                      r'F:\heshuai\proj\matting_tf\dataset_utils\generators\alpha_unique_id.record',
                      r'F:\heshuai\proj\matting_tf\dataset_utils\generators\coco_ful_filter.record',
                      r'F:\heshuai\proj\matting_tf\dataset_utils\generators\manual_ful.record',
                      r'F:\heshuai\proj\matting_tf\dataset_utils\generators\matting-coarse_ful.record'
                      ]
    weights = [1] * len(tf_record_path)
    iterator_train, iterator_val = input_fn_from_tfrecord_binary_test(tf_record_path, batch_size=16, augmentation=1, val_split=True, shape=320, map_cores=4, debug=1)
    idx = 0
    max_num = 1000
    sess = tf.Session()
    while True:
        time_pre = time.time()
        x, y = sess.run(iterator_train)
        print('new batch')
        print(time.time() - time_pre)
        for item_x, item_y in zip(x, y):
            print(idx)
            item_x += 1
            item_x *= 127.5
            item_x = np.uint8(item_x)
            plt.subplot(121)
            plt.imshow(np.squeeze(np.squeeze(item_y[:, :])), 'gray')
            plt.subplot(122)
            plt.imshow(np.squeeze(item_x))
            plt.show()
            # item_x += 1
            # item_x *= 127.5
            # item_y = np.squeeze(item_y)
            # merge = apply_mask(item_x, item_y)
            saved_dir = os.path.join(r'F:\test', str(idx)+'.png')
            plt.savefig(saved_dir)
            # io.imsave(saved_dir, merge)
            if idx > max_num:
                exit(0)
            idx += 1


def parse_single_frame_tf_record(tf_records):
    """
    parse tf records
    :param tf_records:
    :return:
    usage sample:
        test_dir = r'F:\heshuai\proj\matting_tf\dataset_utils\generators\alpha_train_256_0812.record'
        parse_single_frame_tf_record(test_dir)
    """
    def _tf_example_parser(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string)
            })
        image = tf.image.decode_image(features['image'])
        annotation = tf.image.decode_image(features['mask'])
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image_shape = [height, width, 3]
        annotation_shape = [height, width, 1]
        image = tf.reshape(image, image_shape)
        mask = tf.reshape(annotation, annotation_shape)
        # cast dtype
        image = tf.cast(image, dtype=tf.float32)
        mask = tf.cast(mask, dtype=(tf.float32))
        return image, mask
    dataset = tf.data.TFRecordDataset(tf_records)
    # Parse the record into tensors.
    dataset = dataset.map(_tf_example_parser, num_parallel_calls=4)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    with tf.Session() as sess:
        while True:
            img, mask = sess.run(data)
            img = np.uint8(np.squeeze(img))
            mask = np.uint8(np.squeeze(mask))
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(mask, 'gray')
            plt.show()


if __name__ == '__main__':
    test_single_frame()


