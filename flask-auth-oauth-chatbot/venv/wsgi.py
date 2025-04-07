# wsgi.py
from app import create_app

# The Vercel Python runtime executes this file
# and looks for an object called 'app' which is a WSGI application.
app = create_app()

# You generally don't need app.run() here; the WSGI server (like Gunicorn, used by Vercel) handles it.