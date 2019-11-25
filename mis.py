import logging
import telebot
from telebot import apihelper
from threading import Thread
from queue import Queue


def line_profile(func):
    """
    逐行profile
    usage:
        from evaluation_utils import line_profile
        @line_profile
        some_fn()
    reference:
        https://github.com/rkern/line_profiler#kernprof
    :param func:
    :return:
    """
    from functools import wraps
    import line_profiler
    prof = line_profiler.LineProfiler()
    @wraps(func)
    def newfunc(*args, **kwargs):
        try:
            pfunc = prof(func)
            return pfunc(*args, **kwargs)
        finally:
            prof.print_stats(1e-3)
    return newfunc


class MyLog(object):
    """
    requirement: telebot
        https://github.com/eternnoir/pyTelegramBotAPI
    """
    def __init__(self, log_file='output.log', tele_bot_token=None, chat_id=None, clean_format=False):
        # create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.log_file = log_file
        ch = logging.FileHandler(self.log_file)
        ch.setLevel(logging.DEBUG)
        if not clean_format:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # initialize telegram bot
        _default_token = '664787432:AAFkb3Q_mMXlid29fwWibIUOeaNWvtHSHpg'
        self.token = _default_token if tele_bot_token is None else tele_bot_token
        # bot与你的聊天的chat id，默认是你自己
        _default_chat_id = '786535272'
        # 如果你的网络环境不科学，请自备工具，否则会连不上服务器
        apihelper.proxy = {'https': 'http://www.v5lab.cn:8011'}
        self.chat_id = _default_chat_id if chat_id is None else tele_bot_token
        self.tb = telebot.TeleBot(self.token)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def clear_logfile(self):
        with open(self.log_file, 'w'):
            pass
        return

    def fire_message_via_bot(self, message, chat_id=None):
        """
        向指定的chat id发送文字信息
        注意：由于采用markdown解析，字符串中不能出现markdown的语义符，否则报bad request错误
        :param message:
        :param chat_id:
        :return:
        """
        response = None
        if chat_id is None:
            chat_id = self.chat_id
        if (self.token is None) or (chat_id is None):
            self.logger.warning('token or chat_id is required!')
        else:
            response = self.tb.send_message(chat_id, message, parse_mode='Markdown')
        return response


class PersonalThreading:
    def __init__(self, worker_fn, **kwargs):
        self.out_q = Queue(50)
        self.thread = Thread(target=worker_fn, args=(out_q=se))

    def add_worker(self, worker_fn, input_queue=None, output_queue=None):
        fn_name = worker_fn.func_name


if __name__ == '__main__':
    logger = MyLog('test.log', clean_format=True)
    logger.fire_message_via_bot('oh, hi')

