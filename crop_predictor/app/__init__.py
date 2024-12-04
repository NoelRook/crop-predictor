from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap5
from app.middleware import PrefixMiddleware

# Create the Flask application instance.
application = Flask(__name__)

# Create the Flask application instance.
application.config.from_object(Config)

# Set up the database connection and ORM using SQLAlchemy.
db = SQLAlchemy(application)
migrate = Migrate(application, db)
login = LoginManager(application)
login.login_view = 'login'

# Integrate Bootstrap5 into the application for responsive UI design.
bootstrap = Bootstrap5(application)

# Add middleware to the application for handling custom URL prefixes.
# Set `voc=False` for local deployment (no prefix added).
application.wsgi_app = PrefixMiddleware(application.wsgi_app, voc=False)

# Import application routes, models, and additional utility modules.
from app import routes, models
from app.serverlibrary import filepath

# Within the application context, perform database initialization if needed.
with application.app_context():
    # Check if the database is empty; if so, import data from a CSV file.
    if routes.is_database_None():
         routes.import_csv(filepath)