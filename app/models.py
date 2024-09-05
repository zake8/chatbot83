from datetime import datetime, timezone # is this used?
from typing import Optional
import sqlalchemy as sa
import sqlalchemy.orm as so
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login # is login used?
from sqlalchemy.ext.mutable import MutableList


class User(UserMixin, db.Model):
    id:            so.Mapped[int] = so.mapped_column(primary_key=True)
    username:      so.Mapped[str] = so.mapped_column(sa.String(64), index=True, unique=True)
    email:         so.Mapped[str] = so.mapped_column(sa.String(120), index=True, unique=True)
    password_hash: so.Mapped[Optional[str]] = so.mapped_column(sa.String(256))
    role:          so.Mapped[Optional[str]] = so.mapped_column(sa.String(64), default='regular')
    full_name:     so.Mapped[Optional[str]] = so.mapped_column(sa.String(120), default='')
    phone_number:  so.Mapped[Optional[str]] = so.mapped_column(sa.String(20), nullable=True)
    chatbot:       so.Mapped[Optional[str]] = so.mapped_column(sa.String(20), nullable=True)
    rag_selected:  so.Mapped[Optional[str]] = so.mapped_column(sa.String(35), nullable=True)
    rag_used:      so.Mapped[Optional[str]] = so.mapped_column(sa.String(35), nullable=True)
    model:         so.Mapped[Optional[str]] = so.mapped_column(sa.String(20), nullable=True)
    embed_model:   so.Mapped[Optional[str]] = so.mapped_column(sa.String(20), nullable=True)
    llm_temp:      so.Mapped[Optional[float]] = so.mapped_column()
    llm_api_key:   so.Mapped[Optional[str]] = so.mapped_column(sa.String(46), nullable=True)
    rag_list:      so.Mapped[Optional[list]] = so.mapped_column(MutableList.as_mutable(sa.PickleType), default=[])
    chat_history:  so.Mapped[Optional[list]] = so.mapped_column(MutableList.as_mutable(sa.PickleType), default=[])

# role detail:
# "regular" - default value for new account creations (can access GerBot, VingTsunBot, etc.)
# "administrator" - gets and can use chatbot_commands in UI
# "vts" - needed for access to VTSBot (auto assigned for email @vingtsunsito.com)
# "guest" - auto logged on as if no account specified; limited rights, can change
# "disabled" - denied any login

# Steps to update database with changes made in this file:
# pipenv shell
# flask db migrate -m "some change"
# flask db upgrade

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login.user_loader
def load_user(id):
    return db.session.get(User, int(id))