#!/usr/bin/env python


import logging
logging.basicConfig(level=logging.INFO, 
                    filename='./log.log', 
                    filemode='a', 
                    format='%(asctime)s -%(levelname)s - %(message)s')

# pipenv installs needed: beautifulsoup4
from app.prompts import SUMMARY_TEMPLATE, VTT_TRANSCRIPTION_CORRECTIONS_TEMPLATE
from datetime import datetime
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter # tweaked module name
import os
import re
import requests
import subprocess

### # already imported and used in routes.py - need to import here too?
### from langchain_community.vectorstores import FAISS

### # function is in routes.py - not sure if can will just use that, can import, or should just duplicate
### from routes import get_embedding_func


# Ingestion control; chunk for encoding faiss vector DB, map reduce chunk for larger document summary
my_chunk_size = 300 # chunk_size= and chunk_overlap, what should they be, how do they relate to file size, word/token/letter count?
my_chunk_overlap = 100 # what should overlap % be to retain meaning and search-ability? # https://chunkviz.up.railway.app/
create_summary_on_ingest = True
create_corrections_on_ingest = True
my_map_red_chunk_size = 50000 # This is for map reduce summary, the largest text by character length to try to send # Mixtral-8x7b is a max context size of 32k tokens
my_correction_chunk_size = 6000 # This is for chunking to correction parse; seems to timeout on same size it can summerize... (Going with 1/5.)


def ingest_document(fullragchat_rag_source, rag_source_clue_value, model, fullragchat_embed_model, mkey, query, fullragchat_temp, start_page, end_page):
    logging.info(f'===> Attempting ingestion on "{fullragchat_rag_source}", with page range "{start_page}" to "{end_page}". (All pages if Nones.)')
    answer = ''
    if not os.path.exists(fullragchat_rag_source):
        answer += f'Source document "{fullragchat_rag_source}" not found locally. '
        return answer
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source)
    if not match:
        answer += f'There is no extension found on "{fullragchat_rag_source}"'
        return answer
    rag_ext = match.group(1)
    base_fn = os.path.basename(fullragchat_rag_source) # strip path
    base_fn = base_fn[:-(len(rag_ext)+1)] # strip extension
    if start_page and end_page: # tweak filename to save with pdf page numbers
        base_fn = f'{base_fn}-{start_page}-{end_page}'
    faiss_index_fn = f'{base_fn}.faiss'
    # Check if file to save already exists...
    if os.path.exists(faiss_index_fn): ### This doesn't seem to work, maybe 'cause .faiss is a directory?
        answer += f'{faiss_index_fn} already exists; please delete and then retry. '
        return answer
    # Get text
    rag_text = get_rag_text(
        fullragchat_rag_source=fullragchat_rag_source, 
        query=query, 
        start_page=start_page, 
        end_page=end_page )
    answer += f'Read "{fullragchat_rag_source}". '
    # Prep summary
    if create_summary_on_ingest:
        summary_text_for_output = create_map_reduce_summary(
            to_sum = rag_text, 
            map_red_chunk_size = my_map_red_chunk_size, 
            model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
    else:
        summary_text_for_output = f'No summary created.'
    if create_corrections_on_ingest:
        corrected_vtt_fn = f'{base_fn}_corrected.vtt'
        corrections_text_for_output = create_transcription_corrections(
            to_sum = rag_text, 
            map_red_chunk_size = my_correction_chunk_size, 
            model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
        with open(docs_dir + '/' + corrected_vtt_fn, 'w', encoding="utf8") as file:
            file.write(corrections_text_for_output)
    # Write _loadered.txt to disk
    if rag_ext != 'txt': #don't write out a '_loadered.txt' if input was '.txt'
        txtfile_fn = f'{base_fn}_loadered.txt'
        text_string = ''
        for page_number in range(0, len(rag_text) ):
            text_string += rag_text[page_number].page_content + '\n'
            # LangChain document object is a list, each list item is a dictionary with two keys, 
            # page_content and metadata
        with open(docs_dir + '/' + txtfile_fn, 'a', encoding="utf8") as file: # 'a' = append, create new if none
            if start_page and end_page:
                file.write(f'Specifically PDF pages {start_page} to {end_page} \n')
            file.write(text_string)
        logging.info(f'===> Saved new .txt file, "{txtfile_fn}"')
        answer += f'Wrote "{txtfile_fn}". '
    else:
        txtfile_fn = f'{base_fn}.txt' # still notes in clue_file_text
    # Write FAISS to disk
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=my_chunk_size, chunk_overlap=my_chunk_overlap)
    documents = text_splitter.split_documents(rag_text)
    embeddings = get_embedding_func(fullragchat_embed_model=fullragchat_embed_model, mkey=mkey)
    vector = FAISS.from_documents(documents, embeddings)
    ### Could not load library with AVX2 support due to: ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
    vector.save_local(docs_dir + '/' + faiss_index_fn)
    logging.info(f'===> saved new FAISS, "{faiss_index_fn}"')
    answer += f'Wrote "{faiss_index_fn}". '
    # Write new .cur file
    curfile_fn = f'{base_fn}.cur'
    date_time = datetime.now()
    curfile_content  = f'\nCuration content for HITL use. \n\n'
    curfile_content += f'Date and time      = {date_time.strftime("%Y-%m-%d %H:%M:%S")} \n'
    curfile_content += f'Target document    = {fullragchat_rag_source} \n'
    if start_page and end_page:
        curfile_content += f'PDF pages      = {start_page} to {end_page} \n'
    curfile_content += f'Chunk size         = {my_chunk_size} \n'
    curfile_content += f'Chunk overlap      = {my_chunk_overlap} \n'
    curfile_content += f'Saved FAISS DB     = {faiss_index_fn} \n'
    curfile_content += f'# vectors in DB    = {vector.index.ntotal} \n'
    curfile_content += f'Model/temp DB      = {fullragchat_embed_model} / {fullragchat_temp} \n'
    curfile_content += f'Model/temp LLM = {model} / {fullragchat_temp} \n'
    if create_corrections_on_ingest:
        curfile_content += f'Saved new .vtt file, "{corrected_vtt_fn}" \n'
    ##### curfile_content += f'\n<summary>\n{summary_text_for_output}\n</summary>\n'
    with open(docs_dir + '/' + curfile_fn, 'a', encoding="utf8") as file: # 'a' = append, create new if none
        file.write(curfile_content)
    logging.info(f'===> saved new .cur file, "{curfile_fn}"')
    answer += f'Wrote "{curfile_fn}". '
    # Add name and summary to rag source clue file for LLM to use!
    ##### strip = len(f'{docs_dir}/')
    clue_file_text  = '\n'
    clue_file_text += 'filename = "' + faiss_index_fn + '" \n'
    if start_page and end_page:
        clue_file_text += 'pages = "' + start_page + '" to "' + end_page + '" \n'
    clue_file_text += 'about = """ \n' + summary_text_for_output + ' \n"""\n'
    clue_file_text += '\n'
    with open(rag_source_clue_value, 'a', encoding="utf8") as file: # 'a' = append, file pointer placed at end of file
        file.write(clue_file_text)
    logging.info(f'===> Added new .faiss and summary to "{rag_source_clue_value}"')
    answer += f'Updated "{rag_source_clue_value}". '
    return answer


def get_rag_text(fullragchat_rag_source, query, start_page, end_page): # loads from loader fullragchat_rag_source path/file w/ .txt .html .pdf .vtt or .json 
    # function ignores passed query value
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    rag_ext = match.group(1)
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/json
    if (rag_ext == "txt") or (rag_ext == "vtt"):
        loader = TextLoader(fullragchat_rag_source, encoding="utf8") 
        # ex: /path/filename # not sure utf8 needed here
        # needs beautifulsoup4...
    elif (rag_ext == "html") or (rag_ext == "htm"):
        loader = WebBaseLoader(fullragchat_rag_source) # ex: https://url/file.html
    elif rag_ext == "pdf":
        loader = PyPDFLoader(fullragchat_rag_source) 
    elif rag_ext == "json":
        loader = JSONLoader(file_path=fullragchat_rag_source,
            jq_schema='.',
            text_content=False)
    else:
        return f'Unable to make loader for "{fullragchat_rag_source}"!\n '
    # from https://docs.mistral.ai/guides/basic-RAG/
    docs = loader.load() # docs is a type 'document'...
    if start_page and end_page:
        # Reduce docs to just the desired pages
        docs = docs[ int(start_page) - 1 : int(end_page) ]
    return docs


def create_summary(to_sum, model, mkey, fullragchat_temp):
    from app.routes import large_lang_model # here to avoid circular load
    prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    chain = ( prompt | large_lang_model | StrOutputParser() )
    try:
        summary = chain.invoke(to_sum)
    except Exception as err_mess:
        # returns 'Request size limit exceeded' in the form of "KeyError: 'choices'" in chat_models.py
        logging.error(f'===> Got error: {err_mess} when invoking summary chain')
        summary = f'Got error: "{err_mess}". Likely this is "Request size limit exceeded" in the form of "KeyError: choices" in chat_models.py. Maybe try smaller "map_red_chunk_size".'
    return summary


def create_map_reduce_summary(to_sum, map_red_chunk_size, model, mkey, fullragchat_temp):
    logging.info(f'===> Starting Map Reduce')
    # Map
    summary = ''
    piece_summaries = ''
    text_splitter = CharacterTextSplitter(
        separator="\n", # should be something else?
        chunk_size = map_red_chunk_size, 
        chunk_overlap = max(map_red_chunk_size // 5, 500),
        length_function=len, 
        is_separator_regex=False )
    to_sum_str = ''
    for index in range(0, len(to_sum)):
        to_sum_str += to_sum[index].page_content + '\n'
    pieces = text_splitter.create_documents([to_sum_str])
    num_pieces = len(pieces)
    for piece in pieces:
        piece_summaries += f'\n\n<INDIVIDUAL SUMMARY START>\n'
        individual_summary = create_summary(
            to_sum = piece, 
            model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
        logging.info(f'Individual_summary is: \n{individual_summary}')
        piece_summaries += (individual_summary + '\n')
        piece_summaries += f'\n<INDIVIDUAL SUMMARY END>\n\n'
    # Reduce
    logging.info(f'===> Map Reduce Summary with {num_pieces + 1} LLM inferences (character chunk size of "{map_red_chunk_size}"). ')
    summary += create_summary(
        to_sum = piece_summaries, 
        model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
    return summary


def create_transcription_corrections(to_sum, map_red_chunk_size, model, mkey, fullragchat_temp):
    from app.routes import large_lang_model # here to avoid circular load
    corrections = ''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = map_red_chunk_size, 
        chunk_overlap = 0 )
    pieces = text_splitter.split_documents(to_sum)
    num_pieces = len(pieces)
    prompt = ChatPromptTemplate.from_template(VTT_TRANSCRIPTION_CORRECTIONS_TEMPLATE)
    chain = ( prompt | large_lang_model | StrOutputParser() )
    for piece in pieces:
        content = piece.page_content
        # content_str = str(piece[0].page_content) # gets "TypeError: 'Document' object is not subscriptable"
        corrections += chain.invoke(content)
    logging.info(f'===> Created transcription corrections with {num_pieces} LLM inference(s) (character chunk size of "{map_red_chunk_size}"). ')
    return corrections


def chatbot_command(query, rag_source_clue_value, docs_dir, model, fullragchat_embed_model, mkey, fullragchat_temp):
    answer = ''
    pattern = r'^chatbot_command\.([a-z]+)\(([^)]+)\)$'
    match = re.search(pattern, query)
    if match:
        meth = match.group(1)
        path_filename = match.group(2)
        pattern = r'\.([a-zA-Z]{3,5})$'
        match = re.search(pattern, path_filename) # pulls extension
        if path_filename == 'None' or path_filename == 'none':
            rag_ext = 'None'
        elif match:
            rag_ext = match.group(1)
        else:
            rag_ext = ''
        if (rag_ext != 'None') and (rag_ext != 'pdf') and (rag_ext != 'html') and (rag_ext != 'htm') and (rag_ext != 'txt') and (rag_ext != 'json') and (rag_ext != 'vtt'):
            answer += f'Error: Invalid extension request, "{rag_ext}".'
            return answer
        else:
            if meth == 'listusers':
                users = db.session.scalars(query) ### will not work yet
                answer += f'Users in DB: '
                for u in users:
                    answer += (f'***ID: "{u.id}", role: "{u.role}", username: "{u.username}", full_name: "{u.full_name}", email: "{u.email}", phone_number: "{u.phone_number}" ***')
            # set a user's role:
                # https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iv-database
                # from app import app, db
                # from app.models import User
                # import sqlalchemy as sa
                # app.app_context().push()
                # z = db.session.get(User, 1)
                # z.role = 'admin'
                # db.session.commit()
            elif meth == 'summary': # output to chat only
                answer += f'Summary of "{path_filename}": ' + '\n'
                fullragchat_rag_source = path_filename
                some_text_blob = get_rag_text(
                    fullragchat_rag_source=fullragchat_rag_source, 
                    query=query, 
                    start_page=None, 
                    end_page=None )
                answer += create_summary(
                    to_sum=some_text_blob, 
                    model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
            elif meth == 'corrections':
                if rag_ext != 'vtt':
                    answer += f'Expecting .vtt file to correct.'
                    return answer
                else:
                    fullragchat_rag_source = path_filename
                    base_fn = os.path.basename(fullragchat_rag_source) # strip path
                    base_fn = base_fn[:-(len(rag_ext)+1)] # strip extension
                    curfile_fn = f'{base_fn}.cur'
                    corrected_vtt_fn = f'{base_fn}_corrected.vtt'
                    date_time = datetime.now()
                    rag_text = get_rag_text(
                        fullragchat_rag_source=fullragchat_rag_source, 
                        query=query, 
                        start_page=None, 
                        end_page=None )
                    corrections_text = create_transcription_corrections(
                        to_sum = rag_text, 
                        map_red_chunk_size = my_correction_chunk_size, 
                        model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
                    with open(docs_dir + '/' + corrected_vtt_fn, 'w', encoding="utf8") as file: # 'w' = overwrite the existing content if any
                        file.write(corrections_text)
                    curfile_content  = f'\n\nCuration  content for HITL use. \n\n'
                    curfile_content += f'Date and time      = {date_time.strftime("%Y-%m-%d %H:%M:%S")} \n'
                    curfile_content += f'Target document    = {fullragchat_rag_source} \n'
                    curfile_content += f'Model/temp LLM = {model} / {fullragchat_temp} \n'
                    curfile_content += f'Wrote corrected .vtt file, "{corrected_vtt_fn}" \n'
                    with open(docs_dir + '/' + curfile_fn, 'a', encoding="utf8") as file: # 'a' = append, create new if none
                        file.write(curfile_content)
                    logging.info(f'===> Saved new .vtt file, "{corrected_vtt_fn}", and new/updated .cur file, "{curfile_fn}"')
                    answer += f'Wrote "{corrected_vtt_fn}; wrote/updated "{curfile_fn}". '
            elif meth == 'mapreducesummary': # output to chat only
                answer += f'Map reduce summary of "{path_filename}": ' + '\n'
                fullragchat_rag_source = path_filename
                some_text_blob = get_rag_text(
                    fullragchat_rag_source=fullragchat_rag_source, 
                    query=query, 
                    start_page=None, 
                    end_page=None )
                answer += create_map_reduce_summary(
                    to_sum = some_text_blob, 
                    map_red_chunk_size = my_map_red_chunk_size, 
                    model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
            elif meth == 'ingest': # from web or local - saves X as .faiss (and .txt), w/ .cur file, and adds to rag_source_clue_value
                fullragchat_rag_source = path_filename
                answer += ingest_document(
                    fullragchat_rag_source = fullragchat_rag_source, 
                    rag_source_clue_value = rag_source_clue_value, 
                    model = model, 
                    fullragchat_embed_model = fullragchat_embed_model, 
                    mkey = mkey, 
                    query = query, 
                    fullragchat_temp = fullragchat_temp,
                    start_page = None, 
                    end_page = None )
            elif meth == 'download': # just save from web to local
                local_filename = docs_dir + '/' + os.path.basename(path_filename)
                if os.path.exists(local_filename):
                    answer += f'{local_filename} already exists; please delete and then retry. '
                    return answer
                response = requests.get(path_filename)
                if response.status_code == 200:
                    with open(local_filename, 'wb') as file:
                        file.write(response.content)
                    answer += f'Downloaded {path_filename} and saved as {local_filename}. '
                else:
                    answer += f'Fail to download {path_filename}, status code: {response.status_code} '
            elif meth == 'listfiles': # lists available docs on disk
                logging.info(f'===> Attempting to list and parse directory: "{docs_dir}"')
                extensions = (".faiss")
                answer += f'List of docs in "{docs_dir}" with "{extensions}" extension: '
                for file in os.listdir(docs_dir):
                    if file.endswith(extensions):
                        answer += '"' + file  + '" '
                answer += 'End of list. '
            elif meth == 'listclues': # lists available docs as per clues file
                logging.info(f'===> Attempting to open and parse: "{rag_source_clue_value}"')
                answer += f'List of docs called out in "{rag_source_clue_value}": '
                with open(rag_source_clue_value, 'r', encoding="utf8") as file:
                    clues_blob = file.read()
                clues = clues_blob.split('\n')
                for item in clues:
                    pattern = r'"([^"]+\.faiss)"'
                    match = re.search(pattern, item)
                    if match:
                        filename_with_extension = match.group(1)
                        base_filename, extension = filename_with_extension.rsplit('.', 1)
                        if extension == 'faiss':
                            answer += '"' + filename_with_extension  + '" '
            elif meth == 'delete': ### deletes X (low priority to build)
                # check if file to save already exists
                # delete file
                answer = f'Delete not implemented; just use ssh or WinSCP. '
                # answer = f'Deleted "{path_filename}".'
            elif meth == 'batchingest': # batch ingest from list text file
                if os.path.exists(path_filename):
                    with open(path_filename, 'r', encoding="utf8") as file: # This one doesn't need the encoding="utf8"; maybe something in rag_source_clues.txt?
                        batch_list_str = file.read()
                    batch_list = batch_list_str.split('\n')
                    for item in batch_list:
                        if (item[0:2] == '# ') or (item == '') :
                            pass # skips comments and blank lines
                        else:
                            pattern = r'^([\w./]+),\s*(\d+),\s*(\d+)$'
                            match = re.search(pattern, item)
                            if match: # item is pfn, page, page
                                fullragchat_rag_source = match.group(1)
                                start_page = match.group(2)
                                end_page = match.group(3)
                                answer += ingest_document(
                                    fullragchat_rag_source = fullragchat_rag_source, 
                                    rag_source_clue_value = rag_source_clue_value, 
                                    model=model, 
                                    fullragchat_embed_model=fullragchat_embed_model, 
                                    mkey=mkey, 
                                    query=query, 
                                    fullragchat_temp=fullragchat_temp,
                                    start_page=start_page,
                                    end_page=end_page )
                            else:
                                pattern = r'^([\w./]+)$'
                                match = re.search(pattern, item)
                                if match: # item is pfn only w/ no page numbers
                                    fullragchat_rag_source = match.group(1)
                                    answer += ingest_document(
                                        fullragchat_rag_source = fullragchat_rag_source, 
                                        rag_source_clue_value = rag_source_clue_value, 
                                        model=model, 
                                        fullragchat_embed_model=fullragchat_embed_model,
                                        mkey=mkey, 
                                        query=query, 
                                        fullragchat_temp=fullragchat_temp,
                                        start_page=None,
                                        end_page=None )
                                else:
                                    answer += f'Can not process: "{item}" '
                else:
                    answer += f'Unable to batch from non-existent (local) file: "{path_filename}".'
            elif meth == 'test':
                try:
                    answer += f'docs_dir: "{docs_dir}"; rag_source_clue_value: "{rag_source_clue_value}" '
                except:
                    answer += f'Unable to read/display docs_dir and/or rag_source_clue_value. '
                try:
                    answer += f'os.path.exists(rag_source_clue_value) is "{os.path.exists(rag_source_clue_value)}". '
                except:
                    answer += f'os.path.exists(rag_source_clue_value) crashed! '
                try:
                    with open(rag_source_clue_value, 'r', encoding="utf8") as file:
                        clues_blob = file.read()
                    answer += f'Read "{rag_source_clue_value}". '
                except Exception as e:
                    answer += f'Error: {e} Not able to read "{rag_source_clue_value}"! '
                try:
                    with open(rag_source_clue_value, 'a', encoding="utf8") as file:
                        file.write('\ntest\n')
                    answer += f'Wrote to "{rag_source_clue_value}". '
                except:
                    answer += f'Not able to write to "{rag_source_clue_value}"! '
            elif meth == 'apikey':
                answer += f'API key in use is: "{mkey}"'
            elif meth == 'pwd':
                output = subprocess.check_output(["pwd"])
                answer += output.decode("utf-8")
            elif meth == 'ollamalist':
                output = subprocess.check_output(["ollama", "list"])
                output_str = output.decode("utf-8")
                answer += output_str.replace('\n', '<line-break>')
            else: # Invalid command
                answer += 'Error: Invalid command.'
            return answer