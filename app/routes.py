#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO, 
                    filename='./log.log', 
                    filemode='a', 
                    format='%(asctime)s -%(levelname)s - %(message)s')

from urllib.parse import urlsplit
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
# Session management to store user-specific data. Each user gets a unique session object, 
# and their data is isolated from other users' sessions. Store user-specific data in the session object 
# and access it throughout the user's session.
# Each request is handled by a separate thread, and data stored in the request context is isolated between requests.
import sqlalchemy as sa
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm
from app.models import User
import socket


# Some global variable settings

ntfypost = False # Posts ntfy for some chat Q&A
pop_fullragchat_history_over_num = 10 # Each query and answer is appended seperatly, when len history > this #, pops off first _two_ items
webserver_hostname = socket.gethostname()


### TODO:

### CAPTCHA
### How to view all users and their data? How to set admin role?
### auto save website URL to pdf? Then can ingest nice record of website in pdf.
### agent to check actual website, maybe crawl a few branches?
### how to tune (know, increase, decrease, number of vector returns from faiss match?
### tweak so some chattyness of choose rag llm pass gets into answer, not just filename...
### Ability to load a (small) text file as a rag doc and hit LLM w/ whole thing, no vector query 
### find extra or wrong css reference path and clean up folder - were two css files... Not sure how or when used
### Make new top menu page
### async stream from llm to screen...


# General high level public routes

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='ChatBot83', 
        webserver_hostname=webserver_hostname)


@app.route("/gerbotsamples")
def gerbotsamples():
    return render_template('gerbotsamples.html')


@app.route('/something')
@login_required
def something():
    return render_template('something.html', title='something')


# Bot specific routes

@app.route('/ChatBot83')
@login_required
def ChatBot83():
    logging.info(f'===> Starting ChatBot83!')
    current_user.chatbot = 'ChatBot83'
    current_user.model = 'open-mixtral-8x7b'
    current_user.llm_temp = 0.25
    current_user.embed_model = 'mistral-embed'
    current_user.rag_list = ['None']
    current_user.chat_history = []
    current_user.chat_history.append({'user':current_user.chatbot, 
        'message':'Salutations! I am ChatBot83. Basically just chat with Mistral LLM...'})
    current_user.chat_history.append({'user':current_user.chatbot, 
        'message':'Enter question/statment and hit query button below.'})
    db.session.commit()
    return redirect(url_for('chat'))


# GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessible
# Generative AI "chat" about the gerrystahl.net writings
# Code by Zake Stahl
# https://github.com/zake8/GerryStahlWritings
# March, April 2024
# Based on public/shared APIs and FOSS samples
# Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more
@app.route('/GerBot')
@login_required
def GerBot():
    logging.info(f'===> Starting GerBot!')
    current_user.chatbot = 'GerBot'
    current_user.model = 'open-mixtral-8x7b'
    current_user.llm_temp = 0.25
    current_user.embed_model = 'mistral-embed'
    current_user.rag_list = ['Auto']
    current_user.chat_history = []
    current_user.chat_history.append({'user':current_user.chatbot, 
        'message':"Let's chat about Gerry Stahl's writing."})
    db.session.commit()
    return redirect(url_for('chat'))


@app.route('/VTSBot')
@login_required
def VTSBot():
    logging.info(f'===> Starting VTSBot!')
    current_user.chatbot = 'VTSBot'
    current_user.model = 'open-mixtral-8x7b'
    current_user.llm_temp = 0.25
    current_user.embed_model = 'mistral-embed'
    current_user.rag_list = ['Auto']
    current_user.chat_history = []
    current_user.chat_history.append({'user':current_user.chatbot, 
        'message':"VTSBot at your service; referencing knowledge from a corpus of Ving Tsun Kung Fu video transcriptions."})
    db.session.commit()
    return redirect(url_for('chat'))


# Authentication routes

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
        logging.info(f'=*=*=*> User "{current_user.username}" logged in.')
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logging.info(f'=*=*=*> User "{current_user.username}" logging out.')
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
        logging.info(f'=*=*=*> User "{current_user.username}" registered.')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/user/<username>')
@login_required
def user(username):
    user = db.first_or_404(sa.select(User).where(User.username == username))
    return render_template('user.html', user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.username     = form.username.data
        current_user.email        = form.email.data
        current_user.full_name    = form.full_name.data
        current_user.phone_number = form.phone_number.data
        db.session.commit()
        flash('Your changes have been saved.')
        logging.info(f'=*=*=*> User "{current_user.username}" edited profile.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)


# Chat and LLM functions

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


@app.route('/chat')
@login_required
def chat():
    # Bot initializations redirect here
    logging.info(f'===> Entering chat loop') ### add username and bot name; also model, temp, embedding model
    return render_template('chat.html', title='Chat')


@app.route('/pending', methods=['POST'])
@login_required
def pending():
    # chat.html posts here
    query = request.form['query']
    current_user.chat_history.append({'user':current_user.username, 'message':query})
    logging.info(f'===> Query: {query}') ### add username and bot name
    # set pending message while waiting
    current_user.chat_history.append({'user':'System', 
        'message':'pending - please wait for model inferences - small moving graphic on browser tab should indicate working'}) 
    db.session.commit()
    return render_template('pending.html', title='Pending')


@app.route('/reply')
@login_required
def reply():
    # pending.html refreshes here
    # clear pending message
    if current_user.chat_history:
        current_user.chat_history.pop()
    response = 'Placeholder response.'
    ### chain = ( setup_and_retrieval_choose_rag | prompt_choose_rag | large_lang_model | StrOutputParser() | 
    ###           process_rag | 
    ###           setup_and_retrieval_response | render_video | prompt_response | large_lang_model | StrOutputParser() )
    ### response = chain.invoke(query)
    current_user.chat_history.append({'user':current_user.chatbot, 'message':response})
    logging.info(f'===> Response: {response}') ### add username and bot name

    if ntfypost:
        title = f'{current_user.chatbot} on {webserver_hostname}:'
        mess = f'Query: {query}\nResponse: {response}'
        if current_user.chatbot == f'GerBot':
            requests.post('https://ntfy.sh/GerBotAction', headers={'Title' : title}, data=(mess))
        if current_user.chatbot == f'VTSBot':
            requests.post('https://ntfy.sh/VTSBotAction', headers={'Title' : title}, data=(mess))

    # cleanup chat history memory if getting too long
    if len(current_user.chat_history) > pop_fullragchat_history_over_num:
        current_user.chat_history.pop(0) # pops off oldest message:answer
        current_user.chat_history.pop(0) # pops off oldest message:query

    db.session.commit()
    return render_template('chat.html', title='Chat')


# RAG document management / administration functions

@app.route('/ingestion')
@login_required
def ingestion(pfn):
    pass
    return """
        <h1>chat_video_recall</h1><br>
        <br>
        ingested {pfn}<br>
        """