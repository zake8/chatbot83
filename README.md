# chatbot83
Conversational Retrieval Augmented GenerativeAI (RAG) chatbot with agent selected RAG content, user authentication, multiple private knowledge corpuses, and (future) video montage output.

In dev, any changes / additions to class User do a:
    cd
    pipenv shell
    flask db migrate -m "note"
    flask db upgrade (flask db downgrade = downgrades one revision; flask db downgrade base = database at its initial state)

To run in dev:
    cd
    pipenv shell
    flask run --debug --host 192.168.50.125

To set up prod:
    tune or setup /etc/apache2/sites-available/flask-app.conf
    tune or setup /var/www/chatbot83/middleapp.wsgi
    tune or setup /home/leet/webframe/Pipfile
    run pipenv update to make lock file etc
    copy files into /var/www/chatbot83
        chatbot83.py, .flaskenv, .env (prod should have own unique FLASK_SECRET_KEY)
    copy folders into /var/www/chatbot83
        sample, migrations, ChatBot83, app (but not subfolder __pycache__), instance (just make folder, _don't_ copy chatbot83.db)

    give permissions to www-data ???

    cd /home/leet/webframe
    pipenv shell
    cd /var/www/chatbot83
    flask shell ???
    flask db upgrade (Because this application uses SQLite, the upgrade command will detect that a database does not exist and will create it)
