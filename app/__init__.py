#!/usr/bin/env python

import os
script_directory = os.path.dirname(os.path.abspath(__file__))
if script_directory.startswith('/var/www/'):
    mode = 'prod'
else:
    mode = 'dev'

# pipenv installs:
# flask-migrate
# flask-login
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate # https://pypi.org/project/Flask-Migrate/
from flask_login import LoginManager

if mode == 'prod':
    load_dotenv('/var/www/chatbot83/.env') # absolute path for prod w/ Flask under Apache
else:
    load_dotenv('.env')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot83.db'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY') # If doesn't get key throws error in prod (Apache w/ WSGI) but _not_ on flask --debug

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'

from app import routes, models