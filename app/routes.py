#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO, 
                    filename='./log.log', 
                    filemode='a', 
                    format='%(asctime)s -%(levelname)s - %(message)s')

# pipenv installs:
# flask-sqlalchemy
# python-dotenv
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, ChangePasswordForm
from app.models import User
from dotenv import load_dotenv
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
# Flask session management stores user-specific data; each user gets a unique session object, 
# and their data is isolated from other users' sessions. Each request is handled by a separate thread, 
# and data stored in the request context is isolated between requests.
# https://flask-login.readthedocs.io/en/latest/#
from flask_simple_captcha import CAPTCHA
from urllib.parse import urlsplit
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import socket
import sqlalchemy as sa


### TODO:
### --> functionality to watch videos with corrected subtitles
### --> functionality to view corrected subtitles vtt files (download, edit, return)
### --> write whole render_video()
### --> need more bot_specific_examples
### --> CSS beautification
### --> dark mode option
### - email a link to click to confirm email and proceed w/ registration - sample exists in flask mega tutorial
### - feature to email yourself chat thread or current Q&A
### - add content to each bot's nothing.faiss to detail what the bot is about and some overview of the corpus
### - setup some pre-approval, and post registration approval processes
### - feedback form to report pleasure of use, bugs, or suggestions
### - interesting backgrounds for each bot, or even based on RAG doc
### How to view all users and their data? How to set admin role? (Can do in shell w/ context via DB CLI.)
### test fake_llm re-implemented as a runnable to pass to chains
### auto save website URL to pdf? Then can ingest nice record of website in pdf.
### agent to check actual website, maybe crawl a few branches?
### how to tune (know, increase, decrease, number of vector returns from faiss match? (some note on this in comments)
### tweak so some chattyness of choose rag llm pass gets into answer, not just filename...
### Ability to load a (small) text file as a rag doc and hit LLM w/ whole thing, no vector query 
### async stream from llm to screen...
### keywords search - to use as option instead of vector, or in parallel w/ vector search

### Maybe to try to do again:
### CAPTCHA
###     unable to implement Cloudflare Turnstile "natively" as hyphens in return function broke html or flask or something
###     tried to do Flask-Turnstile, but with this turnstile graphic never appeared and verify was always true
###     have flask-simple-captcha in place but don't love it - want an easier captcha, but not re-captcha
###     # from flask_turnstile import Turnstile # https://github.com/Tech1k/flask-turnstile


# CAPTCHA setup

# https://github.com/cc-d/flask-simple-captcha
CAPTCHA_CONFIG = {
    'SECRET_CAPTCHA_KEY': 'LONG_KEY',
    'CAPTCHA_LENGTH': 5, # try 9 for prod
    'CAPTCHA_DIGITS': False, # try True for prod
    'EXPIRE_SECONDS': 600,
    'CAPTCHA_IMG_FORMAT': 'JPEG',
    'EXCLUDE_VISUALLY_SIMILAR': True,
    'BACKGROUND_COLOR': (95, 87, 110), 
    'TEXT_COLOR': (232, 221, 245), 
    'ONLY_UPPERCASE': True, 
}
SIMPLE_CAPTCHA = CAPTCHA(config=CAPTCHA_CONFIG)
app = SIMPLE_CAPTCHA.init_app(app)


# Some global variable settings

load_dotenv('.env')

# Posts ntfy for some chat Q&A
ntfypost = False 

# Each query and answer is appended seperately,
# when len history > this #, pops off first (oldest) _two_ items
pop_fullragchat_history_over_num = 10 # should be like 26

webserver_hostname = socket.gethostname()



# General high level public routes

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='ChatBot83', 
        webserver_hostname=webserver_hostname)


@app.route("/gerbotsamples")
def gerbotsamples():
    return render_template('gerbotsamples.html')


# Bot specific routes

@app.route('/ChatBot83')
@login_required
def ChatBot83():
    logging.info(f'===> Starting ChatBot83!')
    current_user.chatbot = 'ChatBot83'
    current_user.model = 'open-mixtral-8x7b'
    current_user.embed_model = 'mistral-embed'
    current_user.llm_temp = 0.25
    current_user.llm_api_key = os.getenv('MISTRAL_API_KEY')
    current_user.rag_list = ['None', 'Auto'] + gen_rag_list()
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
    current_user.llm_api_key = os.getenv('MISTRAL_API_KEY')
    current_user.rag_list = ['Auto'] + gen_rag_list()
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
    current_user.llm_api_key = os.getenv('MISTRAL_API_KEY')
    current_user.rag_list = ['Auto'] + gen_rag_list()
    current_user.rag_selected = 'Auto'
    current_user.chat_history = []
    current_user.chat_history.append({
        'user':current_user.chatbot, 
        'message':"VTSBot at your service; referencing knowledge from a corpus of Ving Tsun Kung Fu video transcriptions."})
    db.session.commit()
    return redirect(url_for('chat'))


def gen_rag_list(): # returns list
    fn_list = []
    extensions = (".faiss")
    for file in os.listdir(f'{current_user.chatbot}'):
        if file.endswith(extensions) and file != 'nothing.faiss':
            fn_list.append(file)
    return fn_list


# Authentication routes

@app.route('/captcha_test', methods=['GET','POST'])
def captcha_test():
    if request.method == 'GET':
        new_captcha_dict = SIMPLE_CAPTCHA.create()
        return render_template('captcha_test.html', captcha=new_captcha_dict)
    if request.method == 'POST':
        c_hash = request.form.get('captcha-hash')
        c_text = request.form.get('captcha-text')
        if SIMPLE_CAPTCHA.verify(c_text, c_hash):
            return 'Success!'
        else:
            return 'Failed CAPTCHA...'


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    new_captcha_dict = SIMPLE_CAPTCHA.create()
    if form.validate_on_submit():
        c_hash = request.form.get('captcha-hash')
        c_text = request.form.get('captcha-text')
        if not SIMPLE_CAPTCHA.verify(c_text, c_hash):
            flash('CAPTCHA verification failed; refresh for new CAPTCHA.')
            return render_template('login.html', 
                                    title='Sign In',
                                    form=form, 
                                    captcha=new_captcha_dict)
        else:
            user = db.session.scalar(
                sa.select(User).where(User.username == form.username.data))
            if user is None or not user.check_password(form.password.data):
                flash('Invalid username or password')
                return redirect(url_for('login'))
            login_user(user, remember=form.remember_me.data)
            logging.info(f'=*=*=*> User "{current_user.username}" logged in.')
            if current_user.role == 'blocked':
                logging.info(f'=*=*=*> Blocked user "{current_user.username}" as their role is "{current_user.role}".')
                logout_user()
                flash('Unable to login.')
                return redirect(url_for('login'))
            next_page = request.args.get('next')
            if not next_page or urlsplit(next_page).netloc != '':
                next_page = url_for('index')
            return redirect(next_page)
    elif request.method == 'GET':
        return render_template('login.html', 
                                title='Sign In',
                                form=form, 
                                captcha=new_captcha_dict)
    return render_template('login.html', 
                            title='Sign In', 
                            form=form, 
                            captcha=new_captcha_dict)


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
    new_captcha_dict = SIMPLE_CAPTCHA.create()
    if form.validate_on_submit():
        c_hash = request.form.get('captcha-hash')
        c_text = request.form.get('captcha-text')
        if not SIMPLE_CAPTCHA.verify(c_text, c_hash):
            flash('CAPTCHA verification failed; refresh for new CAPTCHA.')
            return render_template('register.html', 
                                    title='Register',
                                    form=form, 
                                    captcha=new_captcha_dict)
        else:
            user = User(username=form.username.data, 
                        email=form.email.data, 
                        full_name=form.full_name.data, 
                        phone_number=form.phone_number.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Congratulations, you are now a registered user!')
            logging.info(f'=*=*=*> New user registered! full_name="{form.full_name.data}" username="{form.username.data}" email="{form.email.data}" phone #="{form.phone_number.data}"')
            return redirect(url_for('login'))
    elif request.method == 'GET':
        return render_template('register.html', 
                                title='Register',
                                form=form, 
                                captcha=new_captcha_dict)
    return render_template('register.html', 
                            title='Register', 
                            form=form, 
                            captcha=new_captcha_dict)


@app.route('/user/<username>')
@login_required
def user(username):
    user = db.first_or_404(sa.select(User).where(User.username == username))
    return render_template('user.html', 
                            user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    new_captcha_dict = SIMPLE_CAPTCHA.create()
    if form.validate_on_submit():
        c_hash = request.form.get('captcha-hash')
        c_text = request.form.get('captcha-text')
        if not SIMPLE_CAPTCHA.verify(c_text, c_hash):
            flash('CAPTCHA verification failed; refresh for new CAPTCHA.')
            return render_template('edit_profile.html', 
                                    title='Edit Profile',
                                    form=form, 
                                    captcha=new_captcha_dict)
        else:
            current_user.username     = form.username.data
            current_user.email        = form.email.data
            current_user.full_name    = form.full_name.data
            current_user.phone_number = form.phone_number.data
            db.session.commit()
            flash('Your changes have been saved.')
            logging.info(f'=*=*=*> User "{current_user.username}" edited profile')
            return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        return render_template('edit_profile.html', 
                                title='Edit Profile',
                                form=form, 
                                captcha=new_captcha_dict)
    return render_template('edit_profile.html', 
                            title='Edit Profile',
                            form=form, 
                            captcha=new_captcha_dict)


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    new_captcha_dict = SIMPLE_CAPTCHA.create()
    if form.validate_on_submit():
        c_hash = request.form.get('captcha-hash')
        c_text = request.form.get('captcha-text')
        if not SIMPLE_CAPTCHA.verify(c_text, c_hash):
            flash('CAPTCHA verification failed; refresh for new CAPTCHA.')
            return render_template('change_password.html', 
                                    title='Change Password',
                                    form=form, 
                                    captcha=new_captcha_dict)
        else:
            if not check_password_hash(current_user.password_hash, form.password.data):
                flash('Incorrect current/old password - NO changes saved.')
                return redirect(url_for('change_password'))
            else:
                current_user.password_hash = generate_password_hash(form.new_password.data)
                db.session.commit()
                flash('Your changes have been saved.')
                logging.info(f'=*=*=*> User "{current_user.username}" changed password')
                return redirect(url_for('change_password'))
    elif request.method == 'GET':
        return render_template('change_password.html', 
                                title='Change Password',
                                form=form, 
                                captcha=new_captcha_dict)
    return render_template('change_password.html', 
                            title='Change Password',
                            form=form, 
                            captcha=new_captcha_dict)


# Chat and LLM functions

# pipenv installs:
# langchain_community.document_loaders
# langchain-core
# langchain-community
# langchain_community.llms
# langchain_mistralai.chat_models
# langchain-mistralai
# faiss-cpu
from app.prompts import CHATBOT83_TEMPLATE, VTSBOT_TEMPLATE, GERBOT_TEMPLATE, get_filename_inc_list_template, SIMPLE_CHAT_TEMPLATE
from app.tools import chatbot_command
from langchain_community.document_loaders import TextLoader
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

    rag_text_runnable = RunnableLambda(rag_text_function)
    history_runnable =  RunnableLambda(convo_mem_function)

    # string in ==> double-parallel (question, history) out
    retrieval_simple_chat = RunnableParallel({
        "question": RunnablePassthrough(), 
        "history":  history_runnable})
    
    # string in ==> triple-parallel (context, question, history) out
    setup_and_retrieval_choose_rag = RunnableParallel({
        "context":  rag_text_runnable, 
        "question": RunnablePassthrough(), 
        "history":  history_runnable})
    
    # triple-parallel (context, question, history) in ==> prompt for llm out
    prompt_choose_rag = ChatPromptTemplate.from_template(get_filename_inc_list_template(dir_name=current_user.chatbot))
    
    # triple-parallel (context, question, history) in ==> prompt for llm out
    if  current_user.chatbot  == 'GerBot':
        prompt = ChatPromptTemplate.from_template(GERBOT_TEMPLATE)
    elif current_user.chatbot == 'VTSBot':
        prompt = ChatPromptTemplate.from_template(VTSBOT_TEMPLATE)
    elif current_user.chatbot == 'ChatBot83':
        prompt = ChatPromptTemplate.from_template(CHATBOT83_TEMPLATE)
    else:
        logging.error(f'ERROR =*=*=> No prompt template for "{current_user.chatbot}" (has retrieved context)')
        prompt = ChatPromptTemplate.from_template(CHATBOT83_TEMPLATE)
    prompt_response = prompt

    answer = ''
    rag_pfn = f'{current_user.chatbot}/nothing.faiss'
    if current_user.role == 'administrator':
        if query.startswith('chatbot_command.'):
            response = chatbot_command(
                query=query, 
                rag_source_clue_value=f'{current_user.chatbot}/rag_source_clues.txt', 
                docs_dir=f'{current_user.chatbot}/', 
                model = current_user.model, 
                fullragchat_embed_model = current_user.embed_model, 
                mkey = current_user.llm_api_key, 
                fullragchat_temp = current_user.llm_temp)
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
        response = chain.invoke(query)
        current_user.chat_history.append({
            'user':current_user.chatbot, 
            'message':response})
        logging.info(f'===> Simple Chat Response "{response}" from "{current_user.chatbot}" for "{current_user.username}"')
        if len(current_user.chat_history) > pop_fullragchat_history_over_num:
            current_user.chat_history.pop(0) # pops off oldest message:answer
            current_user.chat_history.pop(0) # pops off oldest message:query
        db.session.commit()
        return render_template('chat.html', title='Simple Chat')

    elif current_user.rag_selected == 'Auto':
        get_rag_chain = ( setup_and_retrieval_choose_rag
                        | prompt_choose_rag 
                        | large_lang_model
                        | StrOutputParser() 
                        )
        selected_rag = get_rag_chain.invoke(query)
        # string manipulations to go from selected_rag to rag_pfn
        logging.info(f'===> selected_rag: "{selected_rag}"')
        # Should be "return_filename.faiss" or the like; sometimes LLM is chatty tho
        # Comments from LLM show in log, and in chat if unable to parse
        # Potentially dangerous - load only local known safe files
        ### need to implement this safety check!
        ### if contains http or double wack "//" then set answer = 'illegal faiss source' and break/return
        # sanity check that filename is in filesystem and ends in .faiss follows
        pattern = r'\b[A-Za-z0-9_-]+\.[A-Za-z0-9]{3,5}\b'
        filenames = re.findall(pattern, selected_rag)
        if filenames:
            clean_selected_rag = filenames[( len(filenames) - 1 )] # pulls last filename from multiple hits as LLM might ramble and then present final answer.
            answer += f'Selected document "{clean_selected_rag}"! '
            rag_pfn = f'{current_user.chatbot}/{clean_selected_rag}'
            if not os.path.exists(rag_pfn):
                answer += f'Error; the file does not exist... '
                rag_pfn = f'{current_user.chatbot}/nothing.faiss'
            pattern = r'\.([a-zA-Z]{3,5})$'
            match = re.search(pattern, clean_selected_rag)
            if match:
                rag_ext = match.group(1)
                if rag_ext != 'faiss':
                    answer += f'Error; .FAISS file is required at this point... '
                    rag_pfn = f'{current_user.chatbot}/nothing.faiss'
            else:
                answer += f'Error; no extension found. '
                rag_pfn = f'{current_user.chatbot}/nothing.faiss'
        else:
            answer += f'Error; unable to parse out a filename from "{selected_rag}". '
            rag_pfn = f'{current_user.chatbot}/nothing.faiss'
        
    else: # assumes specific rag doc selected by user from dropdown
        rag_pfn = f'{current_user.chatbot}/{current_user.rag_selected}'
    
    logging.info(f'===> rag_pfn: "{rag_pfn}"')
    
    # string in ==> triple-parallel (context, question, history) out
    # load existing faiss, and use as retriever
    embeddings = get_embedding_func(
        fullragchat_embed_model=current_user.embed_model, 
        mkey=current_user.llm_api_key)
    loaded_vector_db = FAISS.load_local(rag_pfn, embeddings, allow_dangerous_deserialization=True)
    setup_and_retrieval_response = RunnableParallel({
        "context" : loaded_vector_db.as_retriever(),
        "question": RunnablePassthrough(),
        "history" : RunnableLambda(convo_mem_function)})
    # Default 'k' (amount of documents to return) is 4 per https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html
    
    chain = ( setup_and_retrieval_response 
            | render_video 
            | prompt_response 
            | large_lang_model 
            | StrOutputParser() 
            ) 
    
    response = answer + chain.invoke(query)
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


# prompt for LLM in ==> LLM response out
def large_lang_model(query):
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
        large_lang_model = RunnableLambda(fake_llm)
        logging.info(f'===> Using fake_llm...')
    else:
        large_lang_model = None
        logging.error(f'===> No LLM named "{current_user.model}" to use on Ollama or via Mistral API call. ') # dev issue only so no alert to user
    return large_lang_model


def get_embedding_func(fullragchat_embed_model, mkey):
    if fullragchat_embed_model == 'mistral-embed':
        embeddings = MistralAIEmbeddings(
                model = fullragchat_embed_model, 
                mistral_api_key = mkey )
    elif fullragchat_embed_model == 'nomic-embed-text':
        embeddings = OllamaEmbeddings(
                model = fullragchat_embed_model )
    return embeddings


def rag_text_function(query):
    context = ''
    rag_source_clues = f'{current_user.chatbot}/rag_source_clues.txt'
    loader = TextLoader(rag_source_clues, encoding="utf8")
    ##### logging.info(f'rag_text_function query = "{query}"')
    context_list = loader.load()
    ##### logging.info(f'context_list returning: "{context_list}"')
    for item in context_list:
        context += item.page_content # langchain document, not a list
    ##### logging.info(f'rag_text_function returning: "{context}"')
    return context


def convo_mem_function(query):
    history = ''
    # all history except last entry of user:query; avoids LLM seeing query and same query in chat history
    for i in range(len(current_user.chat_history) - 1):
        history += f'{current_user.chat_history[i]["user"]}: {current_user.chat_history[i]["message"]}\n'
    ##### logging.info(f'convo_mem_function query = "{query}"')
    ##### logging.info(f'convo_mem_function returning: "{history}"')
    return history


def render_video(query):
    ### triple parallel too? This step in chain needs to feed off of rag FAISS DB embeded lookup k returns!
    ### search rag index for timecode for vectordb_matches
    ### render clips with captions burned
    ### create montage.mp4
    # https://ffmpeg.org/download.html
    return RunnableParallel({
        "context": RunnablePassthrough(), 
        "question": RunnablePassthrough(), 
        "history": RunnablePassthrough()})


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
