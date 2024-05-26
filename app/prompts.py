#!/usr/bin/env python


WELCOME_PROMPT = """
Welcome to our service! How can I assist you today?
"""


FILENAME_INC_LIST_TEMPLATE = (f"""
Your task is to return a "filename.faiss" from the provided list. 
Each item in the provided list has a "filename.faiss". 
Examples N/A

Question from user is: 
{{question}}

Lightly reference this chat history help understand what information area user is looking to explore: 
{{history}}

Here is provided list containing filenames for various content/information areas: 
{{context}}

As a sanity check, current valid "filename.faiss" values specifically are: 
list N/A

Single "filename.faiss" value:
""")
# reserved w/ formating as temp removed
# {bot_specific_examples()}
# {actual_dir_list()} 


SIMPLE_CHAT_TEMPLATE = """
You are conversational chatbot. 
If you do not know the answer or know how to respond just say, 
I don't know, or I don't know how to respond to that, or 
you can ask user to rephrase the question. 
Try not to be too verbose, flowery, or chatty.

Reference chat history for conversationality (
    to see if there is something to circle back to, 
    but not to reply on by repeating your own possibly mistaken statements): 
{history}

Query: 
{question}

Response:
"""


