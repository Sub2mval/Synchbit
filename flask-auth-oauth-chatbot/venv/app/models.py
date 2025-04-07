from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin # Import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db # We'll create db in __init__.py

# db comes from __init__.py

class User(UserMixin, db.Model): # Inherit from UserMixin
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256)) # Increased length for stronger hashes
    # Basic profile info (optional)
    name = db.Column(db.String(100), nullable=True)
    # Relationship to OAuth credentials
    oauth_credentials = db.relationship('OAuth', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Flask-Login expects a __repr__
    def __repr__(self):
        return f'<User {self.email}>'

# Model to store OAuth credentials linked to a User
class OAuth(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    provider = db.Column(db.String(50), nullable=False)  # e.g., 'google', 'microsoft'
    # Store the token dictionary provided by Flask-Dance/OAuthlib
    token = db.Column(db.JSON, nullable=False)
    # Optional: Store provider-specific user ID or email if needed for lookup
    provider_user_id = db.Column(db.String(256), nullable=True, index=True)
    provider_user_email = db.Column(db.String(120), nullable=True) # Store email fetched from provider

    __table_args__ = (db.UniqueConstraint('user_id', 'provider', name='uq_user_provider'),) # User can only connect one account per provider

    def __repr__(self):
        return f'<OAuth {self.provider} for User {self.user_id}>'