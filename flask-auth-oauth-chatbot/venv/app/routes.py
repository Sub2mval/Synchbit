from flask import render_template, flash, redirect, url_for, request, Blueprint
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import db
from app.models import User
from app.models import DataUpload # Import the new model
from app.forms import TabularUploadForm, TextUploadForm, TextFileUploadForm # Import forms
from app.services import process_tabular_upload, process_text_upload # Import services
from werkzeug.utils import secure_filename # For securing filenames
import os

bp = Blueprint('main', __name__)

@bp.route('/')
@bp.route('/index')
def index():
    # Redirect logged-in users directly to chatbot, otherwise show index/login link
    if current_user.is_authenticated:
        return redirect(url_for('main.chatbot'))
    return render_template('index.html', title='Home')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.chatbot')) # Redirect to chatbot if already logged in

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember_me') is not None

        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash('Invalid email or password', 'danger')
            return redirect(url_for('main.login'))

        login_user(user, remember=remember)
        flash(f'Welcome back, {user.email}!', 'success')

        # Redirect to the page user was trying to access, or chatbot page
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.chatbot') # Default redirect after login
        return redirect(next_page)

    return render_template('login.html', title='Sign In')

@bp.route('/logout')
@login_required # Ensure user is logged in to log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main.chatbot')) # Redirect to chatbot

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')

        if not email or not password or not password2:
             flash('Please fill in all fields.', 'warning')
             return redirect(url_for('main.signup'))

        if password != password2:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('main.signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already registered.', 'warning')
            return redirect(url_for('main.signup'))

        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Congratulations, you are now a registered user! Please log in.', 'success')
        return redirect(url_for('main.login'))

    return render_template('signup.html', title='Sign Up')

@bp.route('/chatbot')
@login_required # Protect this page
def chatbot():
    # This is the placeholder page users land on after login
    # We can pass user info or connection status later
    google_connected = current_user.oauth_credentials.filter_by(provider='google').first() is not None
    microsoft_connected = current_user.oauth_credentials.filter_by(provider='microsoft').first() is not None

    return render_template('chatbot.html', title='Chatbot',
                           google_connected=google_connected,
                           microsoft_connected=microsoft_connected)


@bp.route('/connect')
@login_required
def connect():
    # Page showing buttons to connect Google/Microsoft accounts
    google_connected = current_user.oauth_credentials.filter_by(provider='google').first() is not None
    microsoft_connected = current_user.oauth_credentials.filter_by(provider='microsoft').first() is not None
    return render_template('connect.html', title='Connect Accounts',
                           google_connected=google_connected,
                           microsoft_connected=microsoft_connected)

@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    tabular_form = TabularUploadForm(prefix='tabular') # Use prefixes for distinct forms
    text_form = TextUploadForm(prefix='text')
    text_file_form = TextFileUploadForm(prefix='text_file')

    # Determine which form was submitted based on the submit button's name attribute
    if tabular_form.validate_on_submit() and tabular_form.submit.data:
        file = tabular_form.file.data
        dataset_name = tabular_form.dataset_name.data
        # Secure the filename before using it (though we mainly use dataset_name)
        filename = secure_filename(file.filename)

        # Call the service function to process the file
        # Note: This is synchronous. For large files, use Celery task.
        success, message, upload_id = process_tabular_upload(file, current_user.id, dataset_name)
        if success:
            flash(message, 'success')
        else:
            flash(message, 'danger')
        return redirect(url_for('main.list_uploads')) # Redirect to upload list or back to upload page

    if text_form.validate_on_submit() and text_form.submit.data:
        text_content = text_form.text_content.data
        document_name = text_form.document_name.data

        # Call the service function
        # Note: Synchronous processing. Use Celery for large texts.
        success, message, upload_id = process_text_upload(text_content, current_user.id, document_name)
        if success:
            flash(message, 'success')
        else:
            flash(message, 'danger')
        return redirect(url_for('main.list_uploads'))

    if text_file_form.validate_on_submit() and text_file_form.submit.data:
        file = text_file_form.file.data
        document_name = text_file_form.document_name.data
        filename = secure_filename(file.filename) # Secure filename

        try:
            # Read text content from uploaded file
            text_content = file.read().decode('utf-8') # Assume UTF-8 encoding
            file.close() # Close the file handle

            # Call the service function
            success, message, upload_id = process_text_upload(text_content, current_user.id, document_name or filename)
            if success:
                flash(message, 'success')
            else:
                flash(message, 'danger')
        except Exception as e:
            flash(f"Error reading text file '{filename}': {e}", 'danger')

        return redirect(url_for('main.list_uploads'))


    # Render the page with all forms
    return render_template('upload.html', title='Upload Data',
                           tabular_form=tabular_form,
                           text_form=text_form,
                           text_file_form=text_file_form)


@bp.route('/uploads')
@login_required
def list_uploads():
    """Displays a list of data uploads for the current user."""
    uploads = DataUpload.query.filter_by(user_id=current_user.id).order_by(DataUpload.created_at.desc()).all()
    return render_template('list_uploads.html', title='My Uploads', uploads=uploads)