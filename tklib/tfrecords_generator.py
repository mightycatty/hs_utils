import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import skimage.io as io
from skimage.transform import rescale
import tensorflow as tf
from tqdm import tqdm
import os
from dataset_utils.data_util import get_file_list
import sys
from dataset_utils.data_util import get_valid_sample_seq, get_val_list_from_txt, get_valid_sample_single
import random


# TODO:在tfrecord文件中保存文件名称
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
    def rescale(img, min_d=512):
        """
        保持比例，放缩至最小边为min_d
        :param img:
        :param min_d:
        :return:
        """
        m_d = min(img.shape[:2])
        scale_factor = min_d / float(m_d)
        multi_channel = False if np.ndim(img) == 2 else True
        img = rescale(img, scale_factor, preserve_range=True, multichannel=multi_channel)
        return img

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    @staticmethod
    def get_valid_sample(*args):
        image_folder, label_folder = args
        img, mask = get_valid_sample_single(image_folder, label_folder, return_pair=False)
        # shuffle
        random.seed(100)
        random.shuffle(img)
        random.seed(100)
        random.shuffle(mask)
        return img, mask

    @staticmethod
    def get_unique_id_for_val(img_list, ann_list):
        unique_id_img_list = []
        unique_id_ann_list = []
        name_list = [os.path.basename(item) for item in img_list]
        for img_item, ann_item in zip(img_list, ann_list):
            name_item = os.path.basename(img_item)
            if name_list.count(name_item) == 1:
                unique_id_img_list.append(img_item)
                unique_id_ann_list.append(ann_item)
        import random
        random.seed(1)
        random.shuffle(unique_id_img_list)
        random.seed(1)
        random.shuffle(unique_id_ann_list)
        unique_id_img_list = unique_id_img_list[:1000]
        unique_id_ann_list = unique_id_ann_list[:1000]
        return unique_id_img_list, unique_id_ann_list


class SeqInputTFRecord(TFRecordGeneratorBase):
    """
    读取前后共五帧和中间的mask
    """
    data_key = ['height', 'width', 'mask', 't_0', 't_1', 't_2', 't_3', 't_4']

    @staticmethod
    def get_valid_sample(*args):
        base_dir = args[0]
        samples_list = get_valid_sample_seq(base_dir)
        img_seq = [item[:-1] for item in samples_list]
        mask = [item[-1] for item in samples_list]
        return list(zip(img_seq, mask))

    def read_data_entry(self, filename_pairs, min_size=None):
        """
        读取和resize
        :param filename_pairs:
        :param min_size:
        :return:
        """
        for img_path, annotation_path in filename_pairs:
            img_data_list = []
            # 测试图像是否能正常读取
            try:
                for frame_item in img_path:
                    img = io.imread(frame_item)[:, :, :3]  # 有些png图像具有4个通道或者灰度
                    img_data_list.append(img)
                annotation = io.imread(annotation_path, as_gray=True)
                # dump pairs with uncompilable shape
                if annotation.shape[:2] != img_data_list[0].shape[:2]:
                    raise Exception
            except Exception as e:
                print('unknow error with {}:{}'.format(annotation_path, e))
                continue
            # 限制最大分辨率，降低读取io
            if min_size is not None:
                annotation = self.rescale(annotation, min_size)
                img_data_list = [self.rescale(item, min_size) for item in img_data_list]
            height = annotation.shape[0]
            width = annotation.shape[1]
            # value格式 uint8
            img_data_list = [np.uint8(item) for item in img_data_list]
            if annotation.max() == 1:
                annotation = annotation * 255.
            annotation = np.uint8(annotation)
            annotation = np.expand_dims(annotation, axis=-1)
            data_list = [height, width, annotation] + img_data_list
            data_dict = {}
            for key, data_item in zip(self.data_key, data_list):
                data_dict[key] = data_item
            yield data_dict


def tf_example_parser_for_single_frame(serialized_example):
    """
    解析serialized的tf example，不做其他任何操作
    :param serialized_example:
    :return:
    """
    data_key = ['height', 'width', 'image', 'mask']
    def _output_format(features):
        """
        decode image，并按照需要format输出数据形式
        :param features:
        :return:
        """
        # decode image
        image = tf.image.decode_image(features['image'])
        annotation = tf.image.decode_image(features['mask'])
        # shape and value format
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image_shape = [height, width, 3]
        annotation_shape = [height, width, 1]
        image = tf.reshape(image, image_shape)
        mask = tf.reshape(annotation, annotation_shape)
        image = tf.cast(image, dtype=tf.float32)
        mask = tf.cast(mask, dtype=(tf.float32))
        return image, mask
    features_dict = {}
    for key_item in data_key:
        if key_item in ['height', 'width']:
            features_dict[key_item] = tf.FixedLenFeature([], tf.int64)
        else:
            features_dict[key_item] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(
        serialized_example,
        features=features_dict)
    return _output_format(features)


def tf_example_parser_for_seq_frame(serialized_example):
    """
    解析serialized的tf example，不做其他任何操作
    :param serialized_example:
    :return:
    """
    data_key = ['height', 'width', 'mask', 't_0', 't_1', 't_2', 't_3', 't_4']
    def _output_format(features):
        img_data = []
        for key_item in data_key:
            if 't_' in key_item:
                img_data.append(tf.image.decode_image(features[key_item]))
        image = tf.stack(img_data, axis=0)
        mask = tf.image.decode_image(features['mask'])
        # shape and value format
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image_shape = [5, height, width, 3]
        annotation_shape = [height, width, 1]
        image = tf.reshape(image, image_shape)
        mask = tf.reshape(mask, annotation_shape)
        image = tf.cast(image, dtype=tf.float32)
        mask = tf.cast(mask, dtype=(tf.float32))
        return image, mask
    features_dict = {}
    for key_item in data_key:
        if key_item in ['height', 'width']:
            features_dict[key_item] = tf.FixedLenFeature([], tf.int64)
        else:
            features_dict[key_item] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(
        serialized_example,
        features=features_dict)
    return _output_format(features)


def generate_single_frame_tfrecord():
    """
    生成单帧(image-mask)的tfrecord
    :return:
    """
    txt_file = os.path.join(r'F:\result\stage-1\summary_log', 'stage_1-matting_coarse-modest.log')
    output_name = 'stage_1-matting_coarse-modest'
    base_f = r'F:\heshuai\data\segmentation\ready_for_train\corse\opensource\matting_corse'
    max_num = None
    valid_list = get_val_list_from_txt(txt_file)
    test_tf_generator = TFRecordGeneratorBase()
    resize = int(512 * 1.2)
    test_f = os.path.join(base_f, 'image')
    test_ann = os.path.join(base_f, 'annotation')
    img_list, ann_list = test_tf_generator.get_valid_sample(test_f, test_ann)
    if len(valid_list) > 0:
        img_list = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0] in valid_list, img_list))
        ann_list = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0] in valid_list, ann_list))
    data_zip = list(zip(img_list, ann_list))
    print (output_name)
    test_tf_generator.write_data_to_tfrecord(data_zip, output_name, resize, max_num=max_num)


def generate_seq_frame_tfrecord():
    """
    生成单帧(image-mask)的tfrecord
    :return:
    """
    test_tf_generator = SeqInputTFRecord()
    data_type = 'train'
    output_name = 'alpha_seq_{}_256_0820'.format(data_type)
    resize = 256
    test_f = r'F:\heshuai\data\segmentation\ready_for_train\alpha_mask\08-13\{}'.format(data_type)
    data_zip = test_tf_generator.get_valid_sample(test_f)
    test_tf_generator.write_data_to_tfrecord(data_zip, output_name, resize, max_num=None)


if __name__ == '__main__':
    generate_single_frame_tfrecord()