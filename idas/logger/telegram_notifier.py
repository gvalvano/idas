"""
For the guide refer to: https://www.marcodena.it/blog/telegram-logging-handler-for-python-java-bash/
- - - - - - -
bot page:
https://api.telegram.org/bot[TOKENID]/getUpdates
- - - - - - -
bot info:
curl -X GET https://api.telegram.org/bot[TOKENID]/getMe
- - - - - - -
send message to the bot:
curl -s -X POST https://api.telegram.org/bot[TOKENID]/sendMessage -d chat_id=[ID] -d text="Hello world"

"""

import requests
from logging import Handler, Formatter
import logging
# from time import strftime

TELEGRAM_TOKEN = '[...]'  # 'PUT HERE YOUR TOKEN-ID'
TELEGRAM_CHAT_ID = '[...]'  # 'PUT HERE YOUR CHAT-ID'


class RequestsHandler(Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': log_entry,
            'parse_mode': 'HTML'
        }
        return requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=TELEGRAM_TOKEN),
                             data=payload).content
        

class LogstashFormatter(Formatter):
    def __init__(self):
        super(LogstashFormatter, self).__init__()

    def format(self, record):
        # time = strftime("%d/%m/%Y, %H:%M:%S")
        # return "<b>{datetime}</b>\n{message}".format(datetime=time, message=record.msg)
        return "{message}".format(message=record.msg)


def basic_notifier(logger_name, message, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    handler = RequestsHandler()
    formatter = LogstashFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level)
    logger.info(message)


if __name__ == '__main__':
    l_name = 'trymeApp'
    l_msg = 'We have a problem'
    basic_notifier(l_name, l_msg)
