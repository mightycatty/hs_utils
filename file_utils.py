import cv2
import acapture
import logging
import numpy as np


# TODO: 异步模式
def video_reader(video_dir, loop=False, cvt_format='RGB', *args, **kwargs):
    """
    read webcam or video, return a generator, utilizing acapture to webcam reading performance
    :param video_dir: int or abs_dir of a video file
    :param loop: loop over a video, not works if reading from webcam
    :param cvt_format: yield image format, RGB or BRG
    :param args:
    :param kwargs:
    :return:
    """
    assert cvt_format in ['RGB', 'BGR'], 'invalid cvt format-{RGB/BGR}'
    if type(video_dir) is int:
        cap = acapture.open(video_dir) # RGB
        cvt_format_in = 'RGB'
    else:
        cap = cv2.VideoCapture(video_dir) # BGR
        cvt_format_in = 'BGR'
        # cvt_flag = 1
        # cap.set(cv2.CAP_PROP_FPS, 60)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    error_count = 0
    max_error_num = 5
    while True:
        ret, frame = cap.read()
        if ret:
            if cvt_format_in != cvt_format:
                frame = np.transpose(frame, [2, 1, 0])
            yield frame
        else:
            error_count += 1
            if error_count > max_error_num:
                return
            if loop and (type(video_dir) is str):
                del cap
                cap = cv2.VideoCapture(video_dir)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                cap.release()
                return


class VideoWriter:
    def __init__(self, save_name, fps=30, *args, **kwargs):
        self.save_name = save_name
        self.fps = fps
        self._out_size = None
        self._video_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._video_writer.release()
        finally:
            pass

    def reset(self):
        self._video_writer.release()
        self._video_writer = None

    def _video_writer_init(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(self.save_name, fourcc, self.fps, self._out_size)

    def write_frame(self, image, verbose=False):
        try:
           if not self._video_writer:
               self._out_size = (image.shape[1], image.shape[0])
               self._video_writer_init()
           assert (image.shape[0] == self._out_size[1]) & (image.shape[1] == self._out_size[0]), 'image shape not compilable with video saver shape'
           self._video_writer.write(image)
           if verbose:
               cv2.namedWindow("video_writer", cv2.WINDOW_NORMAL)
               cv2.imshow('video_writer', image)
               cv2.waitKey(1)
        except Exception as e:
            logging.error('write frame error:{}').format(e)
            return False
        return True

    def release(self):
        self._video_writer.release()
