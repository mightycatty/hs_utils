# @Time    : 2020/9/30 12:06
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
from logger import MyLog
import time
import pyautogui
if __name__ == '__main__':
    my_log = MyLog()
    img_file = 'my_screenshot.png'

    while True:
        time.sleep(5)
        img = pyautogui.screenshot(img_file)
        time.sleep(0.5)
        my_log.send_image_via_bot(img_file)
        print('send!')