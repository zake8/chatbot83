# ChatBot83

**Conversational Retrieval Augmented GenerativeAI (RAG) chatbot with agent selected RAG content, user authentication, multiple private knowledge corpuses, and (future) video montage output.**


## To run in dev:

- cd where_Pipfile_is
- pipenv shell
- cd where_app_is
- flask run --debug
  - flask run --debug --host IP == to access from other machines instead of localhost only


## For interactive Python

- cd where_Pipfile_is
- pipenv shell
- cd where_app_is
- flask shell (only works if .flaskenv setup as needed and app.py has @app.shell_context_processor)
- python3 (or just use this if can't do flask shell)
  - note will still not be running as www-data
- exit()
- exit


## In dev, for any changes / additions to class User:

- cd where_Pipfile_is
- pipenv shell
- flask db migrate -m "note on changes/additions to class"
- flask db upgrade
  - flask db downgrade == downgrades one revision
  - flask db downgrade base == database at its initial state)


## Then later in prod, for any changes / additions to class User:

- copy in new files / github pull (especially .../migrations/versions/...)
- may have to reset permissions again
- cd where_Pipfile_is
- pipenv shell
- cd where_app_is
- flask db upgrade


## To set up prod:

[ ] Ubuntu v22

[ ] sudo apt-get update

[ ] sudo apt-get upgrade -y

[ ] sudo apt install apache2 -y

[ ] sudo apt install libapache2-mod-wsgi-py3 -y

[ ] sudo apt install ffmpeg -y == only needed on GUI-less server OS where not installed already

[ ] sudo apt install pipenv -y == may throw errors on Ubuntu and reqr workaround
- workaround:
- sudo apt remove pipenv
- pip3 install pipenv
- python3 -m pipenv shell
- pipenv install # this command run inside venv prompt; then "exit" to exit venv
- need to add to your path in /bashrc and refresh it!

[ ]  export PYTHONIOENCODING=utf-8

- commands to copy (and remove as needed) directories and their contents; "-r" is recursive: 
- cp -r source_directory destination_directory
- rm -r directory_name

[ ] (do this last to cut over once all below is working) tune or setup /etc/apache2/sites-available/chatbot8-ssl.conf
- cd /etc/apache2/sites-available
- sudo a2ensite chatbot8-ssl.conf == Apache2 enable site
  - sudo a2dissite chatbot8-ssl.conf == disable a site
- sudo systemctl reload apache2

[ ] tune or setup /var/www/chatbot83/middleapp.wsgi

[ ] tune or setup /home/leet/webframe/Pipfile == this is where "activate_this" in middleapp.wsgi points

[ ] run pipenv install (no other parameters, with Pipfile only, no lock yet, should install everything)
- run pipenv update, as needed to make lock file etc, as may have to pipenv uninstall, install, sync, for some libraries
- pipenv --venv == returns path for use in /etc/apache2/sites-available/middleapp.wsgi
- to really cleanup venv:
  - pipenv --rm
  - rm Pipfile.lock
- pipenv run pip list == see whats installed in venv

[ ] copy files into /var/www/chatbot83
- chatbot83.py, .flaskenv, .env (prod should have own unique FLASK_SECRET_KEY, LLM API keys)

[ ] copy folders into /var/www/chatbot83
- sample, migrations, ChatBot83, app (but not subfolder __pycache__), instance (just make folder, _don't_ copy dev .db)

[ ] copy in any bot folders and set permissions on them

[ ] give permissions to www-data
- assuming a group www-rwx established with root and www-data as members
  - sudo groupadd www-rwx
  - sudo usermod -aG www-rwx root
  - sudo usermod -aG www-rwx www-data
  - groups root
  - groups www-data
- sudo chown -R :www-rwx /path/to/directory
  - changes group to www-rwx for all files and directories within /path/to/directory.
- sudo chmod -R 2775 /path/to/directory
  - sets the permissions to rwxrwsr-x (2775) for all files and directories within /path/to/directory
  - "2" is (sticky) setgid bit so future files/folders created by www-data will be right too
- sudo find /path/to/directory -type d -exec chmod 2775 {} +
  - this command likely not needed
  - explicitly set the setgid bit on directories to ensure new files and directories inherit the group ownership
- set all this up before copying files and folders in and they should all get permissions, or apply to all after in place

[ ] instantiate .db - from migrations files
- cd /home/leet/webframe
- pipenv shell
- cd /var/www/chatbot83
- flask db upgrade (Because this application uses SQLite, the upgrade command will detect that a database does not exist and will create it)
- if/when db changes/expands in dev, a new 'flask db migrate -m "note"' will be run, the migration file copied to prod, and 'flask db upgrade' run again
- ensure after creating/migrating DB, that the .db file is read-only for www-data

[ ] prod tweaks
- clean up to just desired in /var/www/chatbot83/app/templates/index.html
- set routes.py's mode var to 'prod'
- set path for .env in app/__init__.py
