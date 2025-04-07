from flask import render_template, flash, redirect, url_for, request, Blueprint
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import db
from app.models import User

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