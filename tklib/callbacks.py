"""
custom callbacks for tensorflow keras
ready for model debug and testing
"""
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
# from __future__ import absolute_import
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.summary import summary as tf_summary

from .post_training_utils import freeze_keras_model_to_pb


class IntermediateOutputVisualization(Callback):
    """
    将中间层结果可视化到tensorflow当中
    """

    def __init__(self,
                 x,
                 log_dir,
                 y=None,
                 interested_tensor=None
                 ):
        super(IntermediateOutputVisualization, self).__init__()
        self.x = x
        self.y = y
        self.log_dir = log_dir
        self.writer = tf_summary.FileWriter(self.log_dir)
        self.data_pairs = self._read_data_pairs()
        if interested_layers is not None:
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
        middle_img_vis = IntermediateOutputVisualization.make_image(numpy_image)
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, image=middle_img_vis)])
        self.writer.add_summary(summary, epoch)
        return

    # TODO:可视化指定层
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


class CustomModelCheckpoint(Callback):
    """
    modify the behavior of official checkpoint callback
    to save single-gpu model copy under multi-gpus training circumstance.
    Other behavior identical to official one
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 include_optimizer=False,
                 period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.include_opt = include_optimizer
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
                    model_s.save(filepath, overwrite=True, include_optimizer=self.include_opt
                                 )


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
     redirect training info to a telegram bot
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
            print(message)
            print(result)
        except Exception as e:
            print('error with Telegram bot callback:{}'.format(e))


class AbnormalWeightCheck(Callback):
    """
    raise warning if extreme weights are detected every n epoch.
    result logged into a file named as the model
    """

    def __init__(self, log_dir,
                 log_name=None,
                 warning_threshold=1e5,  # abs(value) > threshold
                 # or
                 # abs(value) < [1 / (threshold)]
                 warning_factor=0.,  # mini fraction of abnormal values in a weight metric to raise warning
                 epoch_period=10,  # detection step/epoch
                 verbose=True):
        super(AbnormalWeightCheck, self).__init__()
        self.log_dir = log_dir
        self.epoch_period = epoch_period
        self._verbose = verbose
        self.warning_threshold = warning_threshold
        self.warning_factor = warning_factor
        self.log_name = log_name
        self.logger = None

    def _init_logger(self):
        if self.log_name is None:
            self.log_name = self.model.name
        self.logger = logging.getLogger(self.log_name)
        logging_level = logging.INFO if self._verbose else logging.ERROR
        self.logger.setLevel(logging_level)
        self.log_file = self.log_name + '_abnormal_weights.log'
        self.log_file = os.path.join(self.log_dir, self.log_file)
        ch = logging.FileHandler(self.log_file)
        ch.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def weight_analyse(self):
        """
        detect weights with extreme value
        :param model:
        :return:
        """
        warning_dict = {}
        for item in self.model.weights:
            weights_value = K.get_value(item)
            w_name = item.name
            abnormal_factor = (np.sum(np.abs(weights_value) > self.warning_threshold) + \
                               np.sum(np.abs(weights_value) < (1. / self.warning_threshold)))
            if abnormal_factor > 0:
                warning_dict[w_name] = abnormal_factor / float(weights_value.size)
        # warning_dict = sorted(warning_dict.items(), key=warning_dict.values)
        return warning_dict

    def on_epoch_end(self, epoch, logs=None):
        if self.logger is None:
            self._init_logger()
        if epoch % self.epoch_period == 0:
            warning_dict = self.weight_analyse()
            self.logger.warning(warning_dict)


class ExportFrozenPb(Callback):
    """
    export keras model to a frozen.pb for inference
    """

    def __init__(self, export_dir,
                 export_period=100,
                 optimize_graph=True,
                 verbose=True):
        super(ExportFrozenPb, self).__init__()
        self.export_dir = export_dir
        self.opt_graph = optimize_graph
        self.step = export_period
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            try:
                model_s = self.model.get_layer('model')  # extract single gpu model copy
            except ValueError:
                model_s = self.model
            export_name = model_s.name + '_epoch_{}'.format(epoch)
            freeze_keras_model_to_pb(model_s, self.export_dir, export_name, self.opt_graph, self.verbose)
