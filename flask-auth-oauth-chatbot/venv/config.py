import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env')) # Load .env file

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-should-really-change-this'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Ensure instance folder exists
    INSTANCE_FOLDER_PATH = os.path.join(basedir, 'instance')
    if not os.path.exists(INSTANCE_FOLDER_PATH):
        os.makedirs(INSTANCE_FOLDER_PATH)

    # OAuth Credentials from .env
    GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    MICROSOFT_OAUTH_CLIENT_ID = os.environ.get("MICROSOFT_OAUTH_CLIENT_ID")
    MICROSOFT_OAUTH_CLIENT_SECRET = os.environ.get("MICROSOFT_OAUTH_CLIENT_SECRET")

    # Set base URL for Vercel deployment (adjust if needed)
    # For local testing, Flask-Dance usually figures it out.
    # For production, you might need to set these explicitly if auto-detection fails.
    # os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1" # ONLY FOR HTTP LOCAL TESTING, REMOVE FOR PRODUCTION!