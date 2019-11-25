from skimage import io
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class TFRecordGeneratorBase:
    """
    分割，image-mask pair。同时是基础类，其他类从这继承
    继承该类时，按需重写以下：
    1. data_key: feature dict的key
    2. read_data_entry: 读取(x, y) pair，返回dict, key为上述data_key
    3. output_format: 定义最终送入tf.dataset的数据format
    """
    def __init__(self):
        pass

    data_key = ['height', 'width', 'image', 'mask']

    def read_data_entry(self, filename_pairs, min_size=None):
        for img_path, annotation_path in filename_pairs:
            # 测试图像是否能正常读取
            try:
                # 读取x data
                img = io.imread(img_path)[:, :, :3]  # 有些png图像具有4个通道或者灰度
                # 读取y data
                annotation = io.imread(annotation_path, as_gray=True)
                # dump pairs with uncompilable shape
                if annotation.shape[:2] != img.shape[:2]:
                    raise Exception
            except Exception as e:
                print('unknow error with {}:{}'.format(annotation_path, e))
                continue
            # 限制最大分辨率，降低读取io
            if min_size is not None:
                if min(img.shape[:2]) > min_size:
                    annotation = self.rescale(annotation, min_size)
                    img = self.rescale(img, min_size)
            height = img.shape[0]
            width = img.shape[1]
            # value格式 uint8
            img = np.uint8(img)
            if annotation.max() == 1:
                annotation = annotation * 255.
            annotation = np.uint8(annotation)
            annotation = np.expand_dims(annotation, axis=-1)
            data_list = [height, width, img, annotation]
            data_dict = {}
            for key, data_item in zip(self.data_key, data_list):
                data_dict[key] = data_item
            yield data_dict

    def tf_example_wrapper(self, data_dict):
        """
        将data dict打包成tf example
        :param data_dict:
        :return:
        """
        feature_dict = {}
        for key, data in data_dict.items():
            if key in ['height', 'width']:
                feature_item = self._int64_feature(data)
            else:
                data = tf.image.encode_png(data).numpy()
                feature_item = self._bytes_feature(data)
            feature_dict[key] = feature_item
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def write_data_to_tfrecord(self, filename_pairs, tfrecords_filename, min_size=None, max_num=None):
        tfrecords_filename += '.record'
        tf.enable_eager_execution()
        tf.executing_eagerly()
        bar = tqdm(total=len(filename_pairs))
        num_count = 0
        read_data_generator = self.read_data_entry(filename_pairs, min_size)
        with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:
            for idx in range(len(filename_pairs)):
                try:
                    data_dict = next(read_data_generator)
                except StopIteration:
                    exit()
                example_item = self.tf_example_wrapper(data_dict)
                writer.write(example_item.SerializeToString())
                # 用以debug,最大写入多少个example
                if max_num is not None:
                    if num_count > max_num:
                        break
                num_count += 1
                bar.update()


    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def input_fn_from_tfrecord(tfrecord_filenames_list, batch_size=32, shape=256, resize_method='force',
                                  augmentation=False, augment_ratio=0.3, map_cores=4, val_split=False, val_split_num=1000, debug=False):
    def _mask_shape_fix(image, mask):
        image = tf.reshape(image, (shape, shape, 3))
        mask = tf.reshape(mask, (shape, shape, 1))
        return image, mask
    if type(tfrecord_filenames_list) is str:
        tfrecord_filenames_list = [tfrecord_filenames_list]
    # read data from tf record
    num_parallel_reads = len(tfrecord_filenames_list)
    dataset = tf.data.TFRecordDataset(tfrecord_filenames_list, num_parallel_reads=num_parallel_reads,
                                      buffer_size=None)
    # dataset = dataset.shuffle(buffer_size=1000)
    if val_split:
        val_dataset = dataset.take(val_split_num * num_parallel_reads)
        train_dataset = dataset.skip(val_split_num * num_parallel_reads)
        # ==================================== train dataset ==============================================
        # train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.prefetch(buffer_size=batch_size * 2)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(tf_example_parser_for_single_frame, num_parallel_calls=map_cores)
        if augmentation:
            train_dataset = train_dataset.map(
                lambda img, mask: tuple(tf.py_function(
                    partial(seg_aug_pipeline, random_crop_size=shape, sometimes_per=augment_ratio),
                    [img, mask],
                    [img.dtype, mask.dtype])),
                num_parallel_calls=map_cores)
        train_dataset = train_dataset.map(
            lambda img, mask: tuple(tf.py_function(
                partial(size_value_norm, size=shape, resize_method=resize_method),
                [img, mask],
                [img.dtype, mask.dtype])),
            num_parallel_calls=map_cores)
        train_dataset = train_dataset.map(_mask_shape_fix)
        train_dataset = train_dataset.batch(batch_size)
        if debug:
            iterator = train_dataset.make_one_shot_iterator()
            train_dataset = iterator.get_next()
        # ==================================== val dataset ==============================================
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.map(tf_example_parser_for_single_frame, num_parallel_calls=map_cores)
        val_dataset = val_dataset.map(
            lambda img, mask: tuple(tf.py_function(
                partial(size_value_norm, size=shape, resize_method=resize_method),
                [img, mask],
                [img.dtype, mask.dtype])),
            num_parallel_calls=map_cores)
        val_dataset = val_dataset.map(_mask_shape_fix)
        val_dataset = val_dataset.batch(batch_size)
        if debug:
            iterator = val_dataset.make_one_shot_iterator()
            val_dataset = iterator.get_next()
        return train_dataset, val_dataset
    else:
        # ==================================== train dataset ==============================================
        train_dataset = dataset.prefetch(buffer_size=batch_size * 2)
        # avoid shuffle to stabilize val metrics

        train_dataset = train_dataset.repeat()

        # Parse the record into tensors.
        train_dataset = train_dataset.map(tf_example_parser_for_single_frame, num_parallel_calls=map_cores)
        # augmentation
        if augmentation:
            train_dataset = train_dataset.map(
                lambda img, mask: tuple(tf.py_function(
                    partial(seg_aug_pipeline, random_crop_size=shape, sometimes_per=augment_ratio),
                    [img, mask],
                    [img.dtype, mask.dtype])),
                num_parallel_calls=map_cores)
        # size/value norm
        train_dataset = train_dataset.map(
            lambda img, mask: tuple(tf.py_function(
                partial(size_value_norm, size=shape, resize_method=resize_method),
                [img, mask],
                [img.dtype, mask.dtype])),
            num_parallel_calls=map_cores)
        train_dataset = train_dataset.map(_mask_shape_fix)

        train_dataset = train_dataset.batch(batch_size)
        if debug:
            # tf keras不用进行iterator封装
            iterator = train_dataset.make_one_shot_iterator()
            train_dataset = iterator.get_next()
        return train_dataset