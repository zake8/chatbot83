from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate # https://pypi.org/project/Flask-Migrate/
from flask_login import LoginManager
import os


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot83.db'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'

from app import routes, models