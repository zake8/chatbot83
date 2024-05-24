#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO, 
                    filename='./log.log', 
                    filemode='a', 
                    format='%(asctime)s -%(levelname)s - %(message)s')

from urllib.parse import urlsplit
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
import sqlalchemy as sa
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User
import socket


ntfypost = False # Posts ntfy for some chat Q&A


### TODO:
### CAPTCHA
### Test full_name and phone_number to registration - how to view all users and their data? How to set admin role?
### auto save website URL to pdf? Then can ingest nice record of website in pdf.
### agent to check actual website, maybe crawl a few branches?
### how to tune (know, increase, decrease, number of vector returns from faiss match?
### from flask import session # Use Flask's built-in session management to store user-specific data. Each user gets a unique session object, and their data is isolated from other users' sessions. You can store user-specific data in the session object and access it throughout the user's session.
### from flask import request # Use Flask's request context to store data that is specific to a single request. Request context is thread-local, meaning that each request is handled by a separate thread, and data stored in the request context is isolated between requests.
### tweak so some chattyness of choose rag llm pass gets into answer, not just filename...
### Ability to load a (small) text file as a rag doc and hit LLM w/ whole thing, no vector query 
### find extra or wrong css reference path and clean up folder - were two css files... Not sure how or when used
### Make new top menu page
### async stream from llm to screen...


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='ChatBot83', 
        webserver_hostname=socket.gethostname())


@app.route("/gerbotsamples")
def gerbotsamples():
    return render_template('gerbotsamples.html')


@app.route('/something')
@login_required
def something():
    return render_template('something.html', title='something')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


def setup_and_retrieval_choose_rag(user, query, history):
    pass
    return query


def prompt_choose_rag(user, query, history):
    rag = 'nothing.faiss'
    ### based on query, select a rag
    return rag


def setup_and_retrieval_response(user, query, rag, history):
    pass
    return query


def prompt_response(user, query, rag, history):
    response = 'Default response.'
    ### based on query and rag, craft a response
    return response


def large_lang_model(model, temp, stop_words_list):
    return Ollama(
        model = model, 
        temperature = float(temp), 
        stop = stop_words_list, 
        verbose = True )


def process_rag(query):
    pass
    # load document requested by choose rag prompt, or return some error.
    return query


def render_video(user, reg, vectordb_matches):
    ### triple parallel too
    ### search rag index for timecode for vectordb_matches
    ### render clips with captions burned
    ### create montage.mp4
    return None


@app.route('/ingestion')
def ingestion(pfn):
    pass
    return """
        <h1>chat_video_recall</h1><br>
        <br>
        ingested {pfn}<br>
        """


@app.route('/chat')
def chat_video_return():
    query = 'test'
    ### chain = ( setup_and_retrieval_choose_rag | prompt_choose_rag | large_lang_model | StrOutputParser() | 
    ###           process_rag | 
    ###           setup_and_retrieval_response | render_video | prompt_response | large_lang_model | StrOutputParser() )
    ### response = chain.invoke(query)
    return """
        <h1>chat_video_recall</h1><br>
        <br>
        query = '{query}'<br>
        response = '{response}'<br>
        <video>./montage.mp4</video><br>
        """