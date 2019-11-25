"""
custom callbacks
用于训练log\调试\可视化
"""
import os
from tensorflow.python.keras.callbacks import Callback
from dataset_utils.img_proccesing import resize_with_padding_from_dir
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.summary import summary as tf_summary
import tensorflow as tf
from dataset_utils.img_proccesing import resize_img_by_keep_ratio
import random


class ImageTensorboard(Callback):
    def __init__(self,
                 input_data_folder,
                 log_dir,
                 interested_layers=None
                 ):
        super(ImageTensorboard, self).__init__()
        self.input_data_folder = input_data_folder
        self.log_dir = log_dir
        self.writer = tf_summary.FileWriter(self.log_dir)
        self.data_pairs = self._read_data_pairs()
        self.intermediate_model = self._intermediate_model(interested_layers)

    @staticmethod
    def make_image(numpy_img):
        """
        将numpy img封装成一个tf.summary.image实例
        :param numpy_img:
        :return:
        """
        numpy_img = np.squeeze(numpy_img)
        from PIL import Image
        assert np.ndim(numpy_img) < 4, 'error with image summary'
        if np.ndim(numpy_img) == 2:
            channel = 1
        else:
            channel = 3
        height, width, = numpy_img.shape[:2]
        image = Image.fromarray(numpy_img)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

    def write_image_summary(self, numpy_image, tag, epoch):
        """
        将任意的numpy image写入tensorboard
        :param numpy_image:
        :param tag:
        :param epoch:
        :param summary_writer:
        :return:
        """
        middle_img_vis = IntermediateVisualizationTimapAlphaInTensorboard.make_image(numpy_image)
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, image=middle_img_vis)])
        self.writer.add_summary(summary, epoch)
        return

    def _read_data_pairs(self):
        """
        根据要求读取数据，并预处理成模型输入
        :return: list of (x, y) pair ready for model input
        """
        base_dir = self.input_data_folder
        data_pair = []
        # self_input_processing(data_pair_item)
        return data_pair

    def _output_format(self, x, y, y_pred):
        """
        将结果处理成可视化的图，如image还原0-255, mask的一些ndim操作等
        :param x:
        :param y:
        :param y_pred:
        :return: 一个字典，key为tag, value为图片
        """
        output_list = {} # 一个item一张图
        # key_list = ['image', 'y_gt', 'y_pred']
        return output_list
    # TODO:可视化指定曾
    def _intermediate_model(self, interested_layers):
        """
        可视化指定输出层和最终输出，否则仅仅可视化最终输出
        :param interested_layers:
        :return:
        """
        if interested_layers is None:
            intermediate_model = self.model
        else:
            interested_tensors = [self.model.get_layer(item).output for item in interested_layers] + self.model.outputs
            intermediate_model = tf.keras.models.Model(self.model.input, interested_tensors)
        return intermediate_model

    def on_epoch_end(self, epoch, logs=None):
        for idx, item in enumerate(self.data_pairs):
            x, y = item
            y_pred = self.model.predict_on_batch(x)
            format_output = self._output_format(x, y, y_pred)
            for tag, img in format_output.items():
                self.write_image_summary(img, '{}_{}'.format(tag, idx), epoch)
        self.writer.flush()


class BinaryImageTensorboard(ImageTensorboard):
    """
    输出rgb，输出(w,h,2/1)的中间结果可视化
    """
    def _read_data_pairs(self):
        """
        根据要求读取数据，并预处理成模型输入
        :return: list of (x, y) pair ready for model input
        """
        img_f = os.path.join(self.input_data_folder, 'image')
        label_f = os.path.join(self.input_data_folder, 'annotation')
        img_dir_list = [os.path.join(img_f, item) for item in os.listdir(img_f)]
        ann_dir_list = [os.path.join(label_f, item) for item in os.listdir(label_f)]
        img_list = [resize_img_by_keep_ratio(item, resize_max=256, as_gray=False, mode='max').astype(np.float32) for item in
                    img_dir_list]
        img_list = [item / 127.5 - 1 for item in img_list]
        img_list = [np.expand_dims(item, axis=0) for item in img_list]
        ann_list = [resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max').astype(np.float32) for item in ann_dir_list]
        return list(zip(img_list, ann_list))

    def _output_format(self, x, y, y_pred):
        """
        将结果处理成可视化的图，如image还原0-255, mask的一些ndim操作等
        :param x:
        :param y:
        :param y_pred:
        :return: 一个字典，key为tag, value为图片
        """
        # key_list = ['image', 'y_gt', 'y_pred']
        output_dict = {}
        output_dict['image'] = np.squeeze(np.uint8((x+1)*127.5))
        output_dict['y_gt'] = np.squeeze(np.uint8(y/y.max()*255))
        y_pred = np.squeeze(y_pred)
        # 当是连个class分类时，仅可视化前景通道
        if y_pred.shape[-1] == 2:
            y_pred = y_pred[:, :, -1]
        output_dict['y_pred'] = np.squeeze(np.uint8(y_pred*255))
        return output_dict


class SeqBinaryImageTensorboard(ImageTensorboard):
    """
    输入rgb*5，输出(w,h,2/1)的中间结果可视化
    """
    def _read_data_pairs(self):
        """
        读取连续五帧数据，及其对应的mask
        :return:
        """
        def _preprocessing(x, y):
            x = np.stack(x, axis=0)
            x = np.float32(x)
            x = x / 127.5 - 1
            x = np.expand_dims(x, axis=0)
            return x, y
        base_dir = self.input_data_folder
        annotation_folder = os.path.join(base_dir, 'annotation')
        annotation_path_list = [os.path.join(annotation_folder, item) for item in os.listdir(annotation_folder)]
        data_pair = []
        for item in annotation_path_list:
            sample_frames_item = []
            for i in range(0, 5):
                frame_path_item = item.replace('annotation', 'frame_{}'.format(i))
                frame_path_item = frame_path_item.replace('_2.png', '_{}.png'.format(i))
                sample_frames_item.append(
                    resize_img_by_keep_ratio(frame_path_item, resize_max=256, as_gray=False, mode='max'))
            annotation_data_item = resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max')
            x, y = _preprocessing(sample_frames_item, annotation_data_item)
            data_pair.append([x, y])
        return data_pair

    def _output_format(self, x, y, y_pred):
        """
        将结果处理成可视化的图，如image还原0-255, mask的一些ndim操作等
        :param x:
        :param y:
        :param y_pred:
        :return: 一个字典，key为tag, value为图片
        """
        # key_list = ['image', 'y_gt', 'y_pred']
        output_dict = {}
        x = np.squeeze(x)
        x = x[0]
        output_dict['image'] = np.squeeze(np.uint8((x+1)*127.5))
        output_dict['y_gt'] = np.squeeze(np.uint8(y/y.max()*255))
        y_pred = np.squeeze(y_pred)
        # 当是连个class分类时，仅可视化前景通道
        if y_pred.shape[-1] == 2:
            y_pred = y_pred[:, :, -1]
        output_dict['y_pred'] = np.squeeze(np.uint8(y_pred*255))
        return output_dict


class SeqTrimapImageTensorboard(ImageTensorboard):
    """
    输入rgb*5，输出(w,h,2/1)的中间结果可视化
    """
    def _read_data_pairs(self):
        """
        读取连续五帧数据，及其对应的mask
        :return:
        """
        def _preprocessing(x, y):
            x = np.stack(x, axis=0)
            x = np.float32(x)
            x = x / 127.5 - 1
            x = np.expand_dims(x, axis=0)
            return x, y
        base_dir = self.input_data_folder
        annotation_folder = os.path.join(base_dir, 'annotation')
        annotation_path_list = [os.path.join(annotation_folder, item) for item in os.listdir(annotation_folder)]
        data_pair = []
        for item in annotation_path_list:
            sample_frames_item = []
            for i in range(0, 5):
                frame_path_item = item.replace('annotation', 'frame_{}'.format(i))
                frame_path_item = frame_path_item.replace('_2.png', '_{}.png'.format(i))
                sample_frames_item.append(
                    resize_img_by_keep_ratio(frame_path_item, resize_max=256, as_gray=False, mode='max'))
            annotation_data_item = resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max')
            x, y = _preprocessing(sample_frames_item, annotation_data_item)
            data_pair.append([x, y])
        return data_pair

    def _output_format(self, x, y, y_pred):
        """
        将结果处理成可视化的图，如image还原0-255, mask的一些ndim操作等
        :param x:
        :param y:
        :param y_pred:
        :return: 一个字典，key为tag, value为图片
        """
        # key_list = ['image', 'y_gt', 'y_pred']
        output_dict = {}
        x = np.squeeze(x)[2, :, :, :]
        output_dict['image'] = np.squeeze(np.uint8((x + 1) * 127.5))
        output_dict['y_gt'] = np.squeeze(np.uint8(y / y.max() * 255))
        y_pred = np.squeeze(y_pred)
        y_pred_trimap = y_pred[:, :, :3]
        y_pred_trimap = np.squeeze(np.uint8(y_pred_trimap * 255))
        y_pred_alpha = y_pred[:, :, -1]
        y_pred_alpha = np.squeeze(np.uint8(y_pred_alpha * 255))
        output_dict['y_pred_trimap'] = y_pred_trimap
        output_dict['y_pred_alpha'] = y_pred_alpha
        return output_dict


class TwoFramsVis(SeqBinaryImageTensorboard):
    def _read_data_pairs(self):
        """
        读取连续五帧数据，及其对应的mask
        :return:
        """
        def _preprocessing(x, y):
            x = np.stack(x, axis=0)
            x = np.float32(x)
            x = x / 127.5 - 1
            x = np.expand_dims(x, axis=0)
            return x, y
        base_dir = self.input_data_folder
        annotation_folder = os.path.join(base_dir, 'annotation')
        annotation_path_list = [os.path.join(annotation_folder, item) for item in os.listdir(annotation_folder)]
        data_pair = []
        for item in annotation_path_list:
            sample_frames_item = []
            for i in range(0, 5):
                frame_path_item = item.replace('annotation', 'frame_{}'.format(i))
                frame_path_item = frame_path_item.replace('_2.png', '_{}.png'.format(i))
                sample_frames_item.append(
                    resize_img_by_keep_ratio(frame_path_item, resize_max=256, as_gray=False, mode='max'))
            annotation_data_item = resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max')
            img_prev = random.choice(sample_frames_item)
            img_current = sample_frames_item[2]
            x, y = _preprocessing([img_prev, img_current], annotation_data_item)
            data_pair.append([x, y])
        return data_pair


class TrimapAlphaImageTensorboard(ImageTensorboard):
    """
    输入rgb，输出[trimap, alpha]4channel的可视化
    """
    def _read_data_pairs(self):
        """
        根据要求读取数据，并预处理成模型输入
        :return: list of (x, y) pair ready for model input
        """
        img_f = os.path.join(self.input_data_folder, 'image')
        label_f = os.path.join(self.input_data_folder, 'annotation')
        img_dir_list = [os.path.join(img_f, item) for item in os.listdir(img_f)]
        ann_dir_list = [os.path.join(label_f, item) for item in os.listdir(label_f)]
        img_list = [resize_img_by_keep_ratio(item, resize_max=256, as_gray=False, mode='max').astype(np.float32) for item in
                    img_dir_list]
        img_list = [item / 127.5 - 1 for item in img_list]
        img_list = [np.expand_dims(item, axis=0) for item in img_list]
        ann_list = [resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max').astype(np.float32) for item in ann_dir_list]
        return list(zip(img_list, ann_list))

    def _output_format(self, x, y, y_pred):
        """
        将结果处理成可视化的图，如image还原0-255, mask的一些ndim操作等
        :param x:
        :param y:
        :param y_pred:
        :return: 一个字典，key为tag, value为图片
        """
        # key_list = ['image', 'y_gt', 'y_pred']
        output_dict = {}
        output_dict['image'] = np.squeeze(np.uint8((x+1)*127.5))
        output_dict['y_gt'] = np.squeeze(np.uint8(y/y.max()*255))
        y_pred = np.squeeze(y_pred)
        y_pred_trimap = y_pred[:, :, :3]
        y_pred_trimap = np.squeeze(np.uint8(y_pred_trimap*255))
        y_pred_alpha = y_pred[:, :, -1]
        y_pred_alpha = np.squeeze(np.uint8(y_pred_alpha*255))
        output_dict['y_pred_trimap'] = y_pred_trimap
        output_dict['y_pred_alpha'] = y_pred_alpha
        return output_dict


#TODO: bug to fix
# TODO list: 1.增加可视化指定层输出的操作 2. CyclicLR测试
class IntermediateVisualization(Callback):
    """
    针对binary segmentation model的中间可视化
    """
    def __init__(self,
               input_data_folder,
               save_dir,
               input_size=256,
               ):
        super(IntermediateVisualization, self).__init__()
        self.input_data_folder = input_data_folder
        self.save_dir = save_dir
        self.input_shape=input_size

    def on_epoch_end(self, epoch, logs=None):
        img_f = os.path.join(self.input_data_folder, 'image')
        label_f = os.path.join(self.input_data_folder, 'annotation')
        img_dir_list = [os.path.join(img_f, item) for item in os.listdir(img_f)]
        ann_dir_list = [os.path.join(label_f, item) for item in os.listdir(label_f)]
        img_list = [resize_with_padding_from_dir(item, desired_size=self.input_shape).astype(np.float32) for item in img_dir_list]
        img_list = [item/127.5 - 1 for item in img_list]
        ann_list = [resize_with_padding_from_dir(item, self.input_shape).astype(np.float32) for item in ann_dir_list]
        ann_list = [np.uint8(item > 0) for item in ann_list]
        # try exception to avoid OOM error
        try:
            img_batch = np.stack(img_list, axis=0)
            pre_mask = self.model.predict_on_batch(img_batch)
            pre_mask = np.split(pre_mask, pre_mask.shape[0], axis=0)
            pre_mask = [np.squeeze(item) for item in pre_mask]
        except Exception as e:
            print (e)
            pre_mask = []
            for item in img_list:
                img_batch = np.expand_dims(item, axis=0)
                pre_mask_item = self.model.predict_on_batch(img_batch)
                pre_mask.append(np.squeeze(pre_mask_item))
        # stack for visualization
        i = 0
        for img_item, ann_item, pre_item in zip(img_list, ann_list, pre_mask):
            img_item = np.uint8((img_item + 1) * 127.5)
            ann_item = np.stack([ann_item*255]*3, axis=-1).astype('uint8')
            if np.ndim(pre_item) == 2:
                channel = 1
            else:
                channel = pre_item.shape[-1]
            # 只保留最多三个通道
            if channel > 3:
                pre_item = pre_item[:, :, :3]
                channel = pre_item.shape[-1]
            # 通道可视化
            if channel == 2:
                pre_item = np.hstack(np.split(pre_item, 2, axis=-1))
                pre_item = np.stack([pre_item*255]*3, axis=-1).astype('uint8')
                pre_item = np.squeeze(pre_item)
            elif channel == 3:
                pre_item = np.squeeze(pre_item) / pre_item.max() * 255
                pre_item = np.uint8(pre_item)
            elif channel == 1:
                pre_item = np.squeeze(pre_item) / pre_item.max() * 255
                pre_item = np.uint8(pre_item)
                pre_item = np.stack([pre_item]*3, axis=-1)
            else:
                print ('error with output')
                print(pre_item.shape)
                exit(0)
            merge_img = np.hstack([img_item, ann_item, pre_item])
            saved_name = '{}_{}.png'.format(i, time.ctime())
            saved_name = saved_name.replace(' ', '-')
            saved_name = saved_name.replace(':', '-')
            dst = os.path.join(self.save_dir, saved_name)
            plt.imsave(dst, merge_img)
            i += 1


class IntermediateVisualizationTimapAlpha(Callback):
    """
    针对exp22-28的中间可视化
    """
    def __init__(self,
               input_data_folder,
               save_dir,
               input_size=256,
               ):
        super(IntermediateVisualizationTimapAlpha, self).__init__()
        self.input_data_folder = input_data_folder
        self.save_dir = save_dir
        self.input_shape=input_size

    def on_epoch_end(self, epoch, logs=None):
        img_f = os.path.join(self.input_data_folder, 'image')
        label_f = os.path.join(self.input_data_folder, 'annotation')
        img_dir_list = [os.path.join(img_f, item) for item in os.listdir(img_f)]
        ann_dir_list = [os.path.join(label_f, item) for item in os.listdir(label_f)]
        img_list = [resize_with_padding_from_dir(item, desired_size=self.input_shape).astype(np.float32) for item in img_dir_list]
        img_list = [item/127.5 - 1 for item in img_list]
        ann_list = [resize_with_padding_from_dir(item, self.input_shape).astype(np.float32) for item in ann_dir_list]
        ann_list = [np.uint8(item > 0) for item in ann_list]
        # try exception to avoid OOM error
        try:
            img_batch = np.stack(img_list, axis=0)
            pre_mask = self.model.predict_on_batch(img_batch)
            pre_mask = np.split(pre_mask, pre_mask.shape[0], axis=0)
            pre_mask = [np.squeeze(item) for item in pre_mask]
        except Exception as e:
            print (e)
            pre_mask = []
            for item in img_list:
                img_batch = np.expand_dims(item, axis=0)
                pre_mask_item = self.model.predict_on_batch(img_batch)
                pre_mask.append(np.squeeze(pre_mask_item))
        # stack for visualization
        i = 0
        for img_item, ann_item, pre_item in zip(img_list, ann_list, pre_mask):
            img_item = np.uint8((img_item + 1) * 127.5)
            ann_item = np.stack([ann_item*255]*3, axis=-1).astype('uint8')
            trimap = np.squeeze(pre_item[:, :, :3]) * 255
            trimap = np.uint8(trimap)
            alpha = np.squeeze(pre_item[:, :, -1])
            alpha = np.stack([alpha*255]*3, axis=-1)
            alpha = np.uint8(alpha)
            merge_img = np.hstack([img_item, ann_item, trimap, alpha])
            saved_name = '{}_{}.png'.format(i, time.ctime())
            saved_name = saved_name.replace(' ', '-')
            saved_name = saved_name.replace(':', '-')
            dst = os.path.join(self.save_dir, saved_name)
            plt.imsave(dst, merge_img)
            i += 1


class IntermediateVisualizationTimapAlphaInTensorboard(ImageTensorboard):
    def __init__(self,
               input_data_folder,
               log_dir,
               ):
        super(IntermediateVisualizationTimapAlphaInTensorboard, self).__init__()
        self.input_data_folder = input_data_folder
        self.log_dir = log_dir
        self.writer = tf_summary.FileWriter(self.log_dir)
        self.data_pairs = self._read_data_pairs()

    def _read_data_pairs(self):
        """
        读取连续五帧数据，及其对应的mask
        :return:
        """
        base_dir = self.input_data_folder
        annotation_folder = os.path.join(base_dir, 'annotation')
        annotation_path_list = [os.path.join(annotation_folder, item) for item in os.listdir(annotation_folder)]
        data_pair = []
        for item in annotation_path_list:
            sample_frames_item = []
            for i in range(0, 5):
                frame_path_item = item.replace('annotation', 'frame_{}'.format(i))
                frame_path_item = frame_path_item.replace('_2.png', '_{}.png'.format(i))
                sample_frames_item.append(resize_img_by_keep_ratio(frame_path_item, resize_max=256, as_gray=False, mode='max'))
            annotation_data_item = resize_img_by_keep_ratio(item, resize_max=256, as_gray=True, mode='max')
            sample_item = [sample_frames_item, annotation_data_item]
            sample_item = self._input_processing(sample_item)
            data_pair.append(sample_item)
        return data_pair

    def _input_processing(self, data_pair_item):
        frames, mask = data_pair_item
        frame_middle = frames[2]
        frames_avg = np.mean(np.stack(frames, axis=0), axis=0)
        frame_df = np.abs(frame_middle - frames_avg)
        # value norm
        frame_middle = np.float32(frame_middle)
        frame_middle = frame_middle / 127.5 - 1
        frame_df = np.float32(frame_df)
        frame_df = frame_df / (frame_df.max() + 1e-5) * 255.
        frame_df = frame_df / 127.5 - 1
        mask = np.float32(mask)
        mask = mask / (mask.max() + 1e-5)
        # dimension
        feature_item = np.concatenate([frame_middle, frame_df], axis=-1)
        feature_item = np.expand_dims(feature_item, axis=0)
        assert np.ndim(feature_item) == 4, 'input shape error for intermediate visualization'
        return feature_item, mask

    def on_epoch_end(self, epoch, logs=None):
        for idx, item in enumerate(self.data_pairs):
            feature_item, mask = item
            pre_item = self.model.predict_on_batch(feature_item)
            pre_item = np.squeeze(pre_item)
            trimap = np.squeeze(pre_item[:, :, :3]) * 255
            trimap = np.uint8(trimap)
            alpha = np.squeeze(pre_item[:, :, -1])
            alpha = np.stack([alpha * 255] * 3, axis=-1)
            alpha = np.uint8(alpha)
            # middle image summary
            middle_img_vis = (np.squeeze(feature_item[:, :, :, :3]) + 1) * 127.5
            middle_img_vis = np.uint8(middle_img_vis)
            self.write_image_summary(middle_img_vis, 'middle_img_{}'.format(idx), epoch)
            middle_img_vis = (np.squeeze(feature_item[:, :, :, 3:]) + 1) * 127.5
            middle_img_vis = np.uint8(middle_img_vis)
            self.write_image_summary(middle_img_vis, 'diff_{}'.format(idx), epoch)
            self.write_image_summary(trimap, 'trimap_{}'.format(idx), epoch)
            self.write_image_summary(alpha, 'alpha_{}'.format(idx), epoch)
        self.writer.flush()


class CustomModelCheckpoint(Callback):
  """
  从官方的checkpoint得来，无论训练时是多gpu还是单GPU，保存时都为单GPU checkpoint, 且不保存opt:include_optimizer=False
  """
  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               period=1):
    super(CustomModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.epochs_since_last_save = 0

    if mode not in ['auto', 'min', 'max']:
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    try:
        model_s = self.model.get_layer('model')
    except ValueError:
        model_s = self.model
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self.filepath.format(epoch=epoch + 1, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          pass
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                             current, filepath))
            self.best = current
            if self.save_weights_only:
              model_s.save_weights(filepath, overwrite=True)
            else:
              model_s.save(filepath, overwrite=True, include_optimizer=False)
          else:
            if self.verbose > 0:
              print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))
      else:
        if self.verbose > 0:
          print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        if self.save_weights_only:
          model_s.save_weights(filepath, overwrite=True)
        else:
          model_s.save(filepath, overwrite=True, include_optimizer=False)


# TODO:to test
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.

    reference:
    [1] https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class LogLearningRate(Callback):
  """
  每个epoch结束Log一下当前learning rate
  """
  def __init__(self, log_dir):
      super(LogLearningRate, self).__init__()
      self.log_dir = log_dir
      self.writer = tf_summary.FileWriter(self.log_dir)

  def on_epoch_end(self, epoch, logs=None):
      summary = tf_summary.Summary()
      summary_value = summary.value.add()
      lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
      summary_value.simple_value = lr
      summary_value.tag = 'lr'
      self.writer.add_summary(summary, epoch)
      self.writer.flush()


class TelegramBot(Callback):
    """
     通过telegram bot监控训练进程
     """
    def __init__(self, logger):
        super(TelegramBot, self).__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        try:
            model_name = self.model.name
            message = 'Model:{}\n epoch:{}\n'.format(model_name, epoch).replace('_', '-')
            for key, value in logs.items():
                message += '{}:{}\n'.format(key, round(float(value), 2)).replace('_', '-')
            result = self.logger.fire_message_via_bot(message)
            print (message)
            print(result)
        except Exception as e:
            print('error with Telegram bot callback:{}'.format(e))



