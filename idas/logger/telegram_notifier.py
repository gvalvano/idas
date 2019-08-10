#
#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Code inspired by: https://www.marcodena.it/blog/telegram-logging-handler-for-python-java-bash/
For the guide you are invited to read the above post.
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


class RequestsHandler(Handler):

    def __init__(self, token_id, chat_id):
        super().__init__()
        self.token_id = token_id
        self.chat_id = chat_id

    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'chat_id': self.chat_id,
            'text': log_entry,
            'parse_mode': 'HTML'
        }
        return requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=self.token_id),
                             data=payload).content
        

class LogstashFormatter(Formatter):
    def __init__(self):
        super(LogstashFormatter, self).__init__()

    def format(self, record):
        # time = strftime("%d/%m/%Y, %H:%M:%S")
        # return "<b>{datetime}</b>\n{message}".format(datetime=time, message=record.msg)
        return "{message}".format(message=record.msg)


def basic_notifier(logger_name, token_id, chat_id, message, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    handler = RequestsHandler(token_id=token_id, chat_id=chat_id)
    formatter = LogstashFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level)
    logger.info(message)


if __name__ == '__main__':
    l_name = 'trymeApp'
    l_msg = 'We have a problem'
    t_id = 'insert here your token id'
    c_id = 'insert here your chat id'
    basic_notifier(logger_name=l_name, token_id=t_id, chat_id=c_id, message=l_msg)
