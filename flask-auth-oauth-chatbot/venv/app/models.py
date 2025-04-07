from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin # Import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
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
    

class DataUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=True) # Original filename or description
    data_type = db.Column(db.String(50), nullable=False) # 'tabular' or 'text'
    storage_location = db.Column(db.String(255), nullable=True) # e.g., PG table name or Pinecone namespace
    row_count = db.Column(db.Integer, nullable=True) # For tabular
    vector_count = db.Column(db.Integer, nullable=True) # For text
    status = db.Column(db.String(50), default='pending') # pending, processing, completed, failed
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    user = db.relationship('User', backref='data_uploads')

    def __repr__(self):
        return f'<DataUpload {self.id} ({self.data_type}) by User {self.user_id}>'

# Recommendation: Create ONE table in Neon to hold ALL tabular data rows,
# linked by upload_id. Use JSONB for flexibility.
# You'd create this table manually in Neon or using raw SQL via Flask-Migrate.
# Example SQL (run this via psql or add to a migration file):
"""
CREATE TABLE IF NOT EXISTS uploaded_tabular_data (
    id SERIAL PRIMARY KEY,
    upload_id INTEGER NOT NULL REFERENCES data_upload(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    row_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_uploaded_tabular_data_upload_id ON uploaded_tabular_data(upload_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_tabular_data_row_data_gin ON uploaded_tabular_data USING gin(row_data); -- Optional: GIN index for JSONB querying
"""
# We won't create a SQLAlchemy model for `uploaded_tabular_data` here,
# as we'll interact with it using Pandas and raw/text SQL for simplicity with JSONB.