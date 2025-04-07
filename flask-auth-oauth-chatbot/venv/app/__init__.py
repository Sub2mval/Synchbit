import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

# Initialize extensions globally
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'main.login' # Where to redirect if user needs to log in
login_manager.login_message_category = 'info' # Flash message category

# User loader function required by Flask-Login
@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return db.session.get(User, int(user_id)) # Use db.session.get for primary key lookups

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure instance folder exists (needed for SQLite before init_app)
    if not os.path.exists(app.instance_path):
         os.makedirs(app.instance_path)
         print(f"Created instance folder: {app.instance_path}")

    # Initialize extensions with the app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Register Blueprints
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    from app.oauth import blueprint as google_bp # Import OAuth blueprints
    app.register_blueprint(google_bp, url_prefix="/login") # Prefix helps distinguish login types

    from app.oauth import microsoft_blueprint as ms_bp
    app.register_blueprint(ms_bp, url_prefix="/login")

    # Create database tables if they don't exist (useful for SQLite)
    # In production, rely on Flask-Migrate ('flask db upgrade')
    with app.app_context():
        db.create_all()
        print("Database tables checked/created (if using SQLite). Use 'flask db upgrade' for migrations.")

    return app