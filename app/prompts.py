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