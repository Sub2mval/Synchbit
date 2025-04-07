import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env')) # Load .env file

# Choose an embedding model (adjust if using OpenAI, etc.)
# Make sure the dimension matches your Pinecone index!
# all-MiniLM-L6-v2 has dimension 384
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384 # Set according to the model above

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

    # Database - Use the DATABASE_URL from .env for Neon
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Remove SQLite specific instance folder logic if not needed
    # INSTANCE_FOLDER_PATH = ... (remove or comment out)

    # Pinecone Config
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
    PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

    # Embedding Config
    EMBEDDING_MODEL = EMBEDDING_MODEL_NAME
    EMBEDDING_DIM = EMBEDDING_DIMENSION

    # Optional: File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Example: Limit uploads to 16MB
    UPLOAD_EXTENSIONS_TABULAR = ['.csv']
    UPLOAD_EXTENSIONS_TEXT = ['.txt', '.md'] # Add other text formats if needed