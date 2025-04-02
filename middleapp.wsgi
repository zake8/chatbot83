#!/usr/bin/env python

import sys
import socket
import logging

sys.path.insert(0, '/var/www/chatbot83')

activate_this = '/home/Pi3berry/.local/share/virtualenvs/chatbot83-1NeRuMAn/bin/activate_this.py'

with open(activate_this) as file_:
	exec(file_.read(), dict(__file__=activate_this))

from chatbot83 import app as application
