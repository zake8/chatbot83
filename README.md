# chatbot83
Conversational Retrieval Augmented GenerativeAI (RAG) chatbot with agent selected RAG content, user authentication, multiple private knowledge corpuses, and (future) video montage output.

Any changes / additions to class User do a:
    cd
    pipenv shell
    flask db migrate -m "note"
    flask db upgrade

To run in dev:
    cd
    pipenv shell
    flask run --debug --host 192.168.50.125