from flask import flash, url_for, redirect
from flask_login import current_user, login_required
from flask_dance.contrib.google import make_google_blueprint, google # Provides 'google' proxy object
from flask_dance.contrib.microsoft import make_microsoft_blueprint, microsoft # Provides 'microsoft' proxy object
from flask_dance.consumer import oauth_authorized, oauth_error
from sqlalchemy.orm.exc import NoResultFound
from app import db
from app.models import User, OAuth # Import your models

# --- Google OAuth Blueprint ---
# Note: 'offline=True' requests a refresh token for long-term access (important!)
# 'scope' defines the permissions you request. Start minimal, add more as needed.
# Common scopes: openid, email, profile. For Gmail API later: https://mail.google.com/
blueprint = make_google_blueprint(
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        # Add Gmail scopes later IF needed:
        # "https://mail.google.com/" # Full access - be careful!
        # "https://www.googleapis.com/auth/gmail.readonly"
        # "https://www.googleapis.com/auth/gmail.send"
        # "https://www.googleapis.com/auth/gmail.modify"
    ],
    offline=True, # Get refresh token
    redirect_to="main.connect" # Redirect back to connect page after auth (or chatbot?)
)

# --- Microsoft OAuth Blueprint ---
# Scopes for Microsoft Graph API (Outlook/Microsoft 365)
# Common scopes: openid, email, profile, User.Read. For Mail: Mail.ReadWrite, Mail.Send
microsoft_blueprint = make_microsoft_blueprint(
    scope=[
        "openid",
        "email",
        "profile",
        "User.Read", # Basic profile read
        "offline_access", # Request refresh token
        # Add Mail scopes later IF needed:
        # "Mail.ReadWrite",
        # "Mail.Send",
    ],
    redirect_to="main.connect" # Redirect back after auth
)


# --- Signal Handlers for Successful OAuth ---

@oauth_authorized.connect_via(blueprint) # Catches signal from google blueprint
@login_required # Ensure a user is logged in before connecting account
def google_logged_in(blpr, token):
    if not token:
        flash("Failed to log in with Google.", "danger")
        return redirect(url_for("main.connect"))

    # Use the token to get user info from Google
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        msg = "Failed to fetch user info from Google."
        flash(msg, "danger")
        return redirect(url_for("main.connect"))

    google_info = resp.json()
    google_user_id = str(google_info["id"]) # Ensure it's a string
    google_email = google_info.get("email")

    # Find or create the OAuth mapping in our database
    try:
        # Check if this Google account is already linked to *another* user
        existing_oauth = OAuth.query.filter_by(provider=blpr.name, provider_user_id=google_user_id).first()
        if existing_oauth and existing_oauth.user_id != current_user.id:
             flash(f"This Google account is already linked to a different user ({existing_oauth.user.email}). Please use a different Google account or log in as that user.", "danger")
             return redirect(url_for("main.connect"))

        # Find existing OAuth for current user and this provider
        oauth = OAuth.query.filter_by(provider=blpr.name, user_id=current_user.id).one()

    except NoResultFound:
        # No existing OAuth for this user and provider, create new one
        oauth = OAuth(provider=blpr.name, user_id=current_user.id, token=token, provider_user_id=google_user_id, provider_user_email=google_email)
        db.session.add(oauth)
    else:
        # Update existing OAuth token
        oauth.token = token
        oauth.provider_user_email = google_email # Update email in case it changed

    db.session.commit()
    flash("Successfully connected your Google account!", "success")
    # 'redirect_url' is handled by Flask-Dance based on 'redirect_to' in blueprint setup
    return False # Prevent Flask-Dance's default redirect


@oauth_authorized.connect_via(microsoft_blueprint) # Catches signal from microsoft blueprint
@login_required
def microsoft_logged_in(blpr, token):
    if not token:
        flash("Failed to log in with Microsoft.", "danger")
        return redirect(url_for("main.connect"))

    # Use the token to get user info from Microsoft Graph API
    # Ensure you requested 'User.Read' scope
    resp = microsoft.get("/v1.0/me") # Microsoft Graph endpoint for user info
    if not resp.ok:
        msg = "Failed to fetch user info from Microsoft."
        flash(msg, "danger")
        return redirect(url_for("main.connect"))

    ms_info = resp.json()
    ms_user_id = ms_info["id"]
    # Email might be in 'mail' or 'userPrincipalName'
    ms_email = ms_info.get("mail") or ms_info.get("userPrincipalName")

    try:
        # Check if this MS account is already linked to *another* user
        existing_oauth = OAuth.query.filter_by(provider=blpr.name, provider_user_id=ms_user_id).first()
        if existing_oauth and existing_oauth.user_id != current_user.id:
             flash(f"This Microsoft account is already linked to a different user ({existing_oauth.user.email}). Please use a different Microsoft account or log in as that user.", "danger")
             return redirect(url_for("main.connect"))

        oauth = OAuth.query.filter_by(provider=blpr.name, user_id=current_user.id).one()
    except NoResultFound:
        oauth = OAuth(provider=blpr.name, user_id=current_user.id, token=token, provider_user_id=ms_user_id, provider_user_email=ms_email)
        db.session.add(oauth)
    else:
        oauth.token = token
        oauth.provider_user_email = ms_email

    db.session.commit()
    flash("Successfully connected your Microsoft account!", "success")
    return False # Prevent default redirect


# --- Optional: Signal Handler for OAuth Errors ---
@oauth_error.connect # Catch errors from any Flask-Dance blueprint
def oauth_error_handler(blpr, error, error_description=None, error_uri=None):
    # Log the error details
    print(f"OAuth Error from {blpr.name}:")
    print(f"Error: {error}")
    print(f"Description: {error_description}")
    print(f"URI: {error_uri}")

    # Flash a generic message to the user
    flash(f"An error occurred while trying to connect with {blpr.name.capitalize()}. Please try again.", "danger")

    # Redirect user back to the connection page or another appropriate page
    return redirect(url_for("main.connect")) # Or maybe url_for("main.index")