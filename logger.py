import logging
import telebot
from telebot import apihelper
import os


class MyLog(object):
    """
    requirement: telebot
        https://github.com/eternnoir/pyTelegramBotAPI
    """
    def __init__(self, log_name=None, log_dir='.', logging_level=logging.DEBUG, clean_format=True, clear_file=False):
        # create logger
        if log_name is None:
            log_name = __name__
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        self.log_file = os.path.join(log_dir, log_name + '.log')
        ch = logging.FileHandler(self.log_file)
        ch.setLevel(logging_level)
        if not clean_format:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if clear_file:
            self.clear_logfile()
        self.tb = None

    def telegram_bot_init(self, token=None):
        # initialize telegram bot
        _default_token = '664787432:AAFkb3Q_mMXlid29fwWibIUOeaNWvtHSHpg'
        self.token = token if token else _default_token
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

    def fire_message_via_bot(self, message, chat_id=None, token=None):
        """
        向指定的chat id发送文字信息
        注意：由于采用markdown解析，字符串中不能出现markdown的语义符，否则报bad request错误
        :param message:
        :param chat_id:
        :return:
        """
        chat_id = '786535272' if chat_id is None else chat_id
        if self.tb is None:
            if (token is None) or (chat_id is None):
                self.logger.warning('token or chat_id is required!')
            else:
                self.token = token
                self.chat_id = chat_id
                self.tb = telebot.TeleBot(self.token)
        else:
            self.tb.send_message(chat_id, message, parse_mode='Markdown')


