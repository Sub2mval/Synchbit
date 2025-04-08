from flask import render_template, flash, redirect, url_for, request, Blueprint
from datetime import datetime, timezone
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import db
from app.models import User
from app.models import DataUpload # Import the new model
from app.forms import TabularUploadForm, TextUploadForm, TextFileUploadForm # Import forms
from app.services import process_tabular_upload, process_text_upload # Import services
from werkzeug.utils import secure_filename # For securing filenames
import os
from app.models import Conversation, DataUpload, ChatMessage, ProposedWrite # Add new models
from app.langgraph_agent import run_agent_turn, get_neon_engine # Import the agent runner
from flask import jsonify # For potential API endpoints

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
    #microsoft_connected = current_user.oauth_credentials.filter_by(provider='microsoft').first() is not None
    return render_template('connect.html', title='Connect Accounts',
                           google_connected=google_connected)#,
                           #microsoft_connected=microsoft_connected)

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

@bp.route('/conversations')
@login_required
def list_conversations():
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.created_at.desc()).all()
    return render_template('conversations.html', title='My Conversations', conversations=conversations)

@bp.route('/conversations/new', methods=['GET', 'POST'])
@login_required
def new_conversation():
    if request.method == 'POST':
        name = request.form.get('name', 'New Conversation')
        selected_upload_ids = request.form.getlist('upload_ids') # Get list of selected upload IDs

        if not name:
            flash('Conversation name cannot be empty.', 'warning')
            return redirect(url_for('main.new_conversation'))

        new_conv = Conversation(user_id=current_user.id, name=name)

        # Add selected data sources
        if selected_upload_ids:
            sources = DataUpload.query.filter(
                DataUpload.id.in_([int(id) for id in selected_upload_ids]),
                DataUpload.user_id == current_user.id # Ensure user owns the uploads
            ).all()
            new_conv.data_sources.extend(sources)

        db.session.add(new_conv)
        db.session.commit()
        flash(f'Conversation "{name}" created!', 'success')
        return redirect(url_for('main.chat', conv_id=new_conv.id))

    # GET request: Show form to create conversation
    # Fetch user's completed uploads to allow selection
    available_uploads = DataUpload.query.filter_by(user_id=current_user.id, status='completed').order_by(DataUpload.filename).all()
    return render_template('new_conversation.html', title='Start New Conversation', available_uploads=available_uploads)


# --- Chat Interface ---

@bp.route('/conversation/<int:conv_id>', methods=['GET'])
@login_required
def chat(conv_id):
    conversation = db.session.get(Conversation, conv_id)
    if not conversation or conversation.user_id != current_user.id:
        flash('Conversation not found or you do not have access.', 'danger')
        return redirect(url_for('main.list_conversations'))

    # Fetch messages for display (can add pagination later)
    messages = conversation.messages.order_by(ChatMessage.timestamp.asc()).all()
    # Pass data sources for info
    data_sources = conversation.data_sources

    return render_template('chat.html', title=f'Chat: {conversation.name}',
                           conversation=conversation,
                           messages=messages,
                           data_sources=data_sources)

# API endpoint to handle sending a message (called by JS in chat.html)
@bp.route('/conversation/<int:conv_id>/message', methods=['POST'])
@login_required
def send_message(conv_id):
    conversation = db.session.get(Conversation, conv_id)
    if not conversation or conversation.user_id != current_user.id:
        return jsonify({'error': 'Conversation not found or unauthorized'}), 404

    data = request.get_json()
    user_query = data.get('message')

    if not user_query:
        return jsonify({'error': 'Message content is empty'}), 400

    # --- Trigger LangGraph Agent (Synchronous for now, make async later) ---
    # In production, use Celery: run_agent_turn.delay(conv_id, user_query)
    # and handle response delivery via WebSockets/SSE.
    try:
        agent_response_content = run_agent_turn(conv_id, user_query)
        # The run_agent_turn function now saves messages to DB
        return jsonify({'agent_response': agent_response_content})
    except Exception as e:
        print(f"Error running agent turn from route: {e}")
        return jsonify({'error': f'Failed to process message: {e}'}), 500


# --- Write Approval Management ---

@bp.route('/approvals')
@login_required
def list_approvals():
    pending_writes = ProposedWrite.query.filter_by(
        user_id=current_user.id,
        status='pending'
    ).order_by(ProposedWrite.created_at.desc()).all()
    return render_template('approvals.html', title='Pending Approvals', pending_writes=pending_writes)

@bp.route('/approvals/<int:write_id>/<action>', methods=['POST']) # action is 'approve' or 'reject'
@login_required
def handle_approval_action(write_id, action):
    proposal = db.session.get(ProposedWrite, write_id)
    if not proposal or proposal.user_id != current_user.id:
        flash('Proposal not found or unauthorized.', 'danger')
        return redirect(url_for('main.list_approvals'))

    if proposal.status != 'pending':
         flash(f'This proposal has already been {proposal.status}.', 'warning')
         return redirect(url_for('main.list_approvals'))

    if action == 'approve':
        # --- Execute the SQL ---
        sql_to_execute = proposal.proposed_sql
        print(f"User {current_user.id} approving write ID {write_id}. SQL:\n{sql_to_execute}")
        engine = get_neon_engine()
        try:
            with engine.connect() as connection:
                with connection.begin(): # Execute within a transaction
                    connection.execute(text(sql_to_execute))
                    # If successful, update status
                    proposal.status = 'executed' # Mark as executed directly
                    proposal.reviewed_at = datetime.now(timezone.utc)
                    proposal.executed_at = proposal.reviewed_at
                    db.session.commit()
                    flash(f'Proposal {write_id} approved and executed successfully!', 'success')
                    # Add confirmation message to conversation
                    conv_id = proposal.conversation_id
                    conf_msg = ChatMessage(conversation_id=conv_id, role='system', # Use 'system' role for clarity
                                           content=f"Write operation (Proposal ID: {write_id}) was approved and executed by the user.")
                    db.session.add(conf_msg)
                    db.session.commit()

        except Exception as e:
            db.session.rollback() # Rollback status change if execution failed
            print(f"Error executing approved SQL for proposal {write_id}: {e}")
            proposal.status = 'failed' # Mark as failed
            proposal.status_reason = f"Execution Error: {e}"
            proposal.reviewed_at = datetime.now(timezone.utc)
            db.session.commit()
            flash(f'Proposal {write_id} approved, but execution failed: {e}', 'danger')

    elif action == 'reject':
        print(f"User {current_user.id} rejecting write ID {write_id}.")
        proposal.status = 'rejected'
        proposal.reviewed_at = datetime.now(timezone.utc)
        # Optional: Add rejection reason from form
        proposal.status_reason = request.form.get('reason', 'Rejected by user.')
        db.session.commit()
        flash(f'Proposal {write_id} rejected.', 'info')
         # Add confirmation message to conversation
        conv_id = proposal.conversation_id
        conf_msg = ChatMessage(conversation_id=conv_id, role='system',
                               content=f"Write operation (Proposal ID: {write_id}) was rejected by the user. Reason: {proposal.status_reason}")
        db.session.add(conf_msg)
        db.session.commit()

    else:
        flash('Invalid action.', 'danger')

    return redirect(url_for('main.list_approvals'))