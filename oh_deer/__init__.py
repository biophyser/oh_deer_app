from flask import Flask

app = Flask(__name__)
from oh_deer import views
app.jinja_env.globals.update(zip=zip)