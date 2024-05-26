#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO, 
                    filename='./log.log', 
                    filemode='a', 
                    format='%(asctime)s -%(levelname)s - %(message)s')

from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm
from app.models import User
from dotenv import load_dotenv
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
# Flask session management stores user-specific data; each user gets a unique session object, 
# and their data is isolated from other users' sessions. Each request is handled by a separate thread, 
# and data stored in the request context is isolated between requests.
from urllib.parse import urlsplit
import os
import socket
import sqlalchemy as sa


# Some global variable settings

load_dotenv('.env')

# Posts ntfy for some chat Q&A
ntfypost = False 

# Each query and answer is appended seperately,
# when len history > this #, pops off first (oldest) _two_ items
pop_fullragchat_history_over_num = 10 # should be like 26, if two lines (or zero) given in bot init, or odd (25) if only one given

webserver_hostname = socket.gethostname()


### TODO:

### CAPTCHA
### How to view all users and their data? How to set admin role?
### auto save website URL to pdf? Then can ingest nice record of website in pdf.
### agent to check actual website, maybe crawl a few branches?
### how to tune (know, increase, decrease, number of vector returns from faiss match?
### tweak so some chattyness of choose rag llm pass gets into answer, not just filename...
### Ability to load a (small) text file as a rag doc and hit LLM w/ whole thing, no vector query 
### CSS beautification
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
    current_user.embed_model = 'mistral-embed'
    current_user.llm_temp = 0.25
    current_user.llm_api_key = os.getenv('Mistral_API_key')
    current_user.rag_list = ['None', 'Auto']
    current_user.rag_selected = 'None'
    current_user.chat_history = []
    current_user.chat_history.append({
        'user':current_user.chatbot, 
        'message':f'Salutations! I am ChatBot83. Basically just chat with "{current_user.model}" LLM...'})
    current_user.chat_history.append({
        'user':current_user.chatbot, 
        'message':'Enter question/statment and hit query button below.'})
    db.session.commit()
    return redirect(url_for('chat'))


@app.route('/GerBot')
@login_required
def GerBot():
    # GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessible
    # Generative AI "chat" about the gerrystahl.net writings
    # Code by Zake Stahl
    # https://github.com/zake8/GerryStahlWritings
    # March, April 2024
    # Based on public/shared APIs and FOSS samples
    # Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more
    logging.info(f'===> Starting GerBot!')
    current_user.chatbot = 'GerBot'
    current_user.model = 'open-mixtral-8x7b'
    current_user.embed_model = 'mistral-embed'
    current_user.llm_temp = 0.25
    current_user.llm_api_key = 'test'
    current_user.rag_list = ['Auto']
    current_user.rag_selected = 'Auto'
    current_user.chat_history = []
    current_user.chat_history.append({
        'user':current_user.chatbot, 
        'message':"Let's chat about Gerry Stahl's writing."})
    db.session.commit()
    return redirect(url_for('chat'))


@app.route('/VTSBot')
@login_required
def VTSBot():
    logging.info(f'===> Starting VTSBot!')
    current_user.chatbot = 'VTSBot'
    current_user.model = 'open-mixtral-8x7b'
    current_user.embed_model = 'mistral-embed'
    current_user.llm_temp = 0.25
    current_user.llm_api_key = 'test'
    current_user.rag_list = ['Auto']
    current_user.rag_selected = 'Auto'
    current_user.chat_history = []
    current_user.chat_history.append({
        'user':current_user.chatbot, 
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

# pipenv install langchain_community.document_loaders
# pipenv install langchain-core
# pipenv install langchain-community
# pipenv install langchain_community.llms
# pipenv install langchain_mistralai.chat_models
# pipenv install langchain-mistralai
from app.prompts import *
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser # LLM output to human readable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter # tweaked module name


@app.route('/chat')
@login_required
def chat():
    # Bot initializations redirect here
    logging.info(f'===> Entering chat loop for "{current_user.username}" with chatbot "{current_user.chatbot}" running models "{current_user.model}" and "{current_user.embed_model}" at "{current_user.llm_temp}"')
    return render_template('chat.html', title='Chat')


@app.route('/pending', methods=['POST'])
@login_required
def pending():
    # chat.html posts here
    current_user.rag_selected = request.form['rag']
    query = request.form['query']
    current_user.chat_history.append({
        'user':current_user.username, 
        'message':query})
    logging.info(f'===> Query "{query}" from "{current_user.username}" for "{current_user.chatbot}"')
    # set pending message while waiting
    current_user.chat_history.append({
        'user':'System', 
        'message':'Pending - please wait for model inferences - small moving graphic on browser tab should indicate working.'}) 
    db.session.commit()
    return render_template('pending.html', title='Pending')


@app.route('/reply')
@login_required
def reply():
    # pending.html refreshes here
    if current_user.chat_history:
        current_user.chat_history.pop() # clear "system: pending" message
    query = current_user.chat_history[-1]["message"] # pull just the message string from the last dictionary in a list 

    # string in ==> triple-parallel (context, question, history) out
    rag_text_runnable = RunnableLambda(rag_text_function)
    history_runnable =  RunnableLambda(convo_mem_function)
    setup_and_retrieval_choose_rag = RunnableParallel({
        "context":  rag_text_runnable, 
        "question": RunnablePassthrough(), 
        "history":  history_runnable
        })
    
    # string in ==> double-parallel (question, history) out
    retrieval_simple_chat = RunnableParallel({
        "question": RunnablePassthrough(), 
        "history":  history_runnable
        })
    
    # triple-parallel (context, question, history) in ==> prompt for llm out
    prompt_choose_rag = ChatPromptTemplate.from_template(FILENAME_INC_LIST_TEMPLATE)
    logging.info(f'prompt_choose_rag = "{prompt_choose_rag}"; type "{type(prompt_choose_rag)}"') ###
    
    # prompt for LLM in ==> LLM response out
    large_lang_model = get_large_lang_model_func()
    logging.info(f'large_lang_model type is "{type(large_lang_model)}"') ###
    
    if current_user.role == 'administrator':
        if query == f'admin stuff':
            pass ### do admin stuff! section is roughed out only for testing
            response = 'Did admin stuff.'
            current_user.chat_history.append({
                'user':current_user.chatbot, 
                'message':response})
            logging.info(f'===> Response "{response}" from "{current_user.chatbot}" for "{current_user.username}"')
            db.session.commit()
            return render_template('chat.html', title='Admin Mode')
    
    if (current_user.rag_selected == 'None') or (current_user.rag_selected == '') or (current_user.rag_selected == None):
        # double-parallel (question, history) in ==> prompt for llm out
        prompt_simple_chat = ChatPromptTemplate.from_template(SIMPLE_CHAT_TEMPLATE)
        chain = ( retrieval_simple_chat
                | prompt_simple_chat 
                | large_lang_model 
                | StrOutputParser() 
                )
    
    elif current_user.rag_selected == 'Auto':
        chain = ( setup_and_retrieval_choose_rag
                | prompt_choose_rag 
                | large_lang_model
                | StrOutputParser() 
                )
###     works with chain.invoke(query): RunnablePassthrough(); straight = RunnablePassthrough(); def straight_func():, pass, return RunnablePassthrough(); 
###     chain = ( setup_and_retrieval_choose_rag | prompt_choose_rag | large_lang_model | readable | process_rag | setup_and_retrieval_response | render_video | prompt_response | large_lang_model | readable )
    
### else: # assumes specific rag doc selected by user from dropdown
###     chain = ( setup_and_retrieval_response 
###             | render_video 
###             | prompt_response 
###             | large_lang_model 
###             | StrOutputParser() 
###             )
    
    response = chain.invoke(query)
    # httpx.LocalProtocolError: Illegal header value b'Bearer ' means missing API key
    current_user.chat_history.append({
        'user':current_user.chatbot, 
        'message':response})
    logging.info(f'===> Response "{response}" from "{current_user.chatbot}" for "{current_user.username}"')
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
    return render_template('chat.html', title='Chat') # loops back in html


def get_large_lang_model_func():
    logging.info(f'get_large_lang_model_func tracks current_user.model as "{current_user.model}"')
    if ( (current_user.model == "open-mixtral-8x7b") or 
        (current_user.model == "mistral-large-latest") or 
        (current_user.model == "open-mistral-7b") ):
            large_lang_model = ChatMistralAI(
                model_name = current_user.model, 
                mistral_api_key = current_user.llm_api_key, 
                temperature = current_user.llm_temp, 
                verbose = True )
            # https://api.python.langchain.com/en/latest/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html
    elif ( (current_user.model == "orca-mini") or 
        (current_user.model == "phi3") or 
        (current_user.model == "tinyllama") or
        (current_user.model == "llama2") or 
        (current_user.model == "llama2-uncensored") or 
        (current_user.model == "mistral") or 
        (current_user.model == "mixtral") or 
        (current_user.model == "command-r") or 
        (current_user.model == "phi") ):
            large_lang_model = Ollama(
                model = current_user.model, 
                temperature = current_user.llm_temp, 
                verbose = True )
            # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
    elif current_user.model == "fake_llm":
        large_lang_model = RunnableLambda(fake_llm) ### # was answer = fake_llm(query); re-implemented as a runnable to pass to chains
        logging.info(f'===> Using fake_llm...')
    else:
        large_lang_model = None
        logging.error(f'===> No LLM named "{current_user.model}" to use on Ollama or via Mistral API call. ') # dev issue only so no alert to user
    return large_lang_model


def rag_text_function(query):
    context = ''
    rag_source_clues = f'{current_user.chatbot}/rag_source_clues.txt'
    loader = TextLoader(rag_source_clues, encoding="utf8")
    # logging.info(f'rag_text_function query = "{query}"') ###
    context_list = loader.load()
    # logging.info(f'context_list returning: "{context_list}"') ###
    for item in context_list:
        context += item.page_content # langchain document, not a list
    logging.info(f'rag_text_function returning: "{context}"') ###
    return context


def convo_mem_function(query):
    history = ''
    # pop off query from end of history, but don't commit; avoids LLM seeing query and same query in chat history
    if current_user.chat_history:
        current_user.chat_history.pop()
    for line in current_user.chat_history:
        history += f'{line["user"]}: {line["message"]}\n'
    # logging.info(f'convo_mem_function query = "{query}"') ###
    logging.info(f'convo_mem_function returning: "{history}"') ###
    return history


def actual_dir_list():
    fn_list = ''
    extensions = (".faiss")
    for file in os.listdir(f'{current_user.chatbot}'):
        if file.endswith(extensions):
            fn_list += '"' + file  + '", '
    if len(fn_list) > 2:
        fn_list = fn_list[:-2] + '. ' # Change last trailing comma to a period
    return fn_list


def bot_specific_examples():
    ### need bot_specific_examples
    return examples


def setup_and_retrieval_response():
    # load existing faiss, and use as retriever
    # Potentially dangerous - load only local known safe files
    ### need to implement this safety check!
    ### if f'{current_user.chatbot}/' contains http or double wack "//" then set answer = 'illegal faiss source' and break/return
    embeddings = get_embedding_func(fullragchat_embed_model=current_user.embed_model, mkey=current_user.llm_api_key)
    loaded_vector_db = FAISS.load_local(current_user.rag_list, embeddings, allow_dangerous_deserialization=True)
    return RunnableParallel({
        "context" : loaded_vector_db.as_retriever(),
        "question": RunnablePassthrough(),
        "history" : RunnableLambda(convo_mem_function)
        })
    # Default 'k' (amount of documents to return) is 4 per https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html

def prompt_response():
    if  current_user.chatbot  == 'GerBot': prompt = ChatPromptTemplate.from_template(gerbot_template)
    elif current_user.chatbot == 'VTSBot': prompt = ChatPromptTemplate.from_template(vtsbot_template)
    elif current_user.chatbot == 'ChatBot8': prompt = ChatPromptTemplate.from_template(chatbot8_template)
    elif  current_user.chatbot == 'Default': prompt = ChatPromptTemplate.from_template(chatbot8_template) ### need/using default? safety from class?
    else: prompt = ChatPromptTemplate.from_template(chatbot8_template)
    return prompt


def process_rag():
    ### load document requested by choose rag prompt, or return some error.
    logging.info(f'===> selected_rag: "something?"')
    return RunnablePassthrough()


def render_video():
    ### triple parallel too? This step in chain needs to feed off of rag FAISS DB embeded lookup k returns!
    ### search rag index for timecode for vectordb_matches
    ### render clips with captions burned
    ### create montage.mp4
    return RunnablePassthrough()


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