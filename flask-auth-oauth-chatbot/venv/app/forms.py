from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, FileField
from wtforms.validators import DataRequired, Length, ValidationError
from flask_wtf.file import FileRequired, FileAllowed
from flask import current_app

# Helper function for file validation
def file_extension_validator(extensions):
    def _validate(form, field):
        if field.data:
            filename = field.data.filename
            allowed = '.' in filename and \
                      filename.rsplit('.', 1)[1].lower() in [ext.lstrip('.') for ext in extensions]
            if not allowed:
                 raise ValidationError(f"File type not allowed. Please upload: {', '.join(extensions)}")
    return _validate

class TabularUploadForm(FlaskForm):
    dataset_name = StringField('Dataset Name (Optional)', validators=[Length(max=100)])
    file = FileField('CSV File', validators=[
        FileRequired(),
        file_extension_validator(current_app.config['UPLOAD_EXTENSIONS_TABULAR'])
    ])
    submit = SubmitField('Upload Tabular Data')

class TextUploadForm(FlaskForm):
    document_name = StringField('Document Name / Source', validators=[DataRequired(), Length(min=1, max=150)])
    text_content = TextAreaField('Paste Text Content', validators=[DataRequired(), Length(min=10)])
    submit = SubmitField('Upload Text Data')

# Optional: Form for uploading text files directly
class TextFileUploadForm(FlaskForm):
    document_name = StringField('Document Name / Source', validators=[DataRequired(), Length(min=1, max=150)])
    file = FileField('Text File (.txt, .md)', validators=[
        FileRequired(),
        file_extension_validator(current_app.config['UPLOAD_EXTENSIONS_TEXT'])
    ])
    submit = SubmitField('Upload Text File')
Use code with caution.
Python
Step 6: Create Processing Services (app/services.py)

Create a new file app/services.py to hold the core logic for handling uploads.

import os
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text # To execute raw SQL for JSONB inserts
from flask import current_app
from app import db
from app.models import DataUpload, User
import uuid # For unique identifiers if needed

# --- Initialize Clients ---
# Initialize Pinecone (consider doing this within create_app context if needed)
def init_pinecone():
    api_key = current_app.config.get('PINECONE_API_KEY')
    environment = current_app.config.get('PINECONE_ENVIRONMENT')
    if not api_key or not environment:
        print("Pinecone API Key or Environment not configured.")
        return None
    try:
        pinecone.init(api_key=api_key, environment=environment)
        return pinecone
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return None

# Load Embedding Model (cache it)
# Consider loading this once globally or using a caching mechanism
embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        model_name = current_app.config['EMBEDDING_MODEL']
        print(f"Loading embedding model: {model_name}")
        try:
            embedding_model = SentenceTransformer(model_name)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            raise e # Re-raise if model loading fails critically
    return embedding_model

# --- Tabular Data Processing ---
def process_tabular_upload(file_storage, user_id, dataset_name=None):
    """Reads CSV, stores metadata, and inserts rows into Neon DB JSONB table."""
    upload_entry = None
    original_filename = file_storage.filename
    if not dataset_name:
        dataset_name = os.path.splitext(original_filename)[0] # Use filename without extension

    try:
        # 1. Create Metadata Entry (mark as processing)
        upload_entry = DataUpload(
            user_id=user_id,
            filename=dataset_name,
            data_type='tabular',
            storage_location='uploaded_tabular_data', # Name of the target table
            status='processing'
        )
        db.session.add(upload_entry)
        db.session.commit() # Commit to get the upload_id

        # 2. Read CSV data using Pandas
        try:
            df = pd.read_csv(file_storage)
            # Basic validation: Check if dataframe is empty
            if df.empty:
                 raise ValueError("CSV file is empty or could not be parsed.")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}") from e

        num_rows = len(df)
        print(f"Read {num_rows} rows from {original_filename} for upload ID {upload_entry.id}")

        # 3. Insert data into Neon 'uploaded_tabular_data' table row by row
        # Using raw SQL with JSONB is often easiest here
        insert_sql = text("""
            INSERT INTO uploaded_tabular_data (upload_id, row_index, row_data)
            VALUES (:upload_id, :row_index, :row_data::jsonb)
        """)

        for index, row in df.iterrows():
            # Convert row to dictionary, handle potential NaN/NaT values for JSON compatibility
            row_dict = row.where(pd.notnull(row), None).to_dict()
            db.session.execute(insert_sql, {
                'upload_id': upload_entry.id,
                'row_index': index,
                'row_data': json.dumps(row_dict) # Convert dict to JSON string
            })

        # 4. Update Metadata Entry (mark as completed)
        upload_entry.status = 'completed'
        upload_entry.row_count = num_rows
        db.session.commit()
        print(f"Successfully processed tabular upload ID {upload_entry.id}")
        return True, f"Successfully uploaded {num_rows} rows from {original_filename} as '{dataset_name}'.", upload_entry.id

    except Exception as e:
        db.session.rollback() # Rollback any partial inserts on error
        error_msg = f"Failed to process tabular upload '{original_filename}': {e}"
        print(error_msg)
        if upload_entry and upload_entry.id: # Check if entry was created before error
            # Update status to failed if possible
            try:
                entry = db.session.get(DataUpload, upload_entry.id)
                if entry:
                    entry.status = 'failed'
                    entry.error_message = str(e)[:1000] # Limit error message length
                    db.session.commit()
            except Exception as inner_e:
                print(f"Additionally failed to update upload status to failed: {inner_e}")
                db.session.rollback()
        return False, error_msg, None


# --- Text Data Processing ---
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter # Example splitter

def process_text_upload(text_content, user_id, document_name):
    """Chunks text, embeds it, and upserts vectors to Pinecone."""
    upload_entry = None
    pinecone_client = init_pinecone()
    if not pinecone_client:
        return False, "Pinecone client not initialized.", None

    pinecone_index_name = current_app.config['PINECONE_INDEX_NAME']
    if pinecone_index_name not in pinecone_client.list_indexes():
         error_msg = f"Pinecone index '{pinecone_index_name}' does not exist. Please create it first."
         print(error_msg)
         return False, error_msg, None

    pinecone_index = pinecone_client.Index(pinecone_index_name)
    # Use upload_id as namespace for data isolation within the index
    pinecone_namespace = None

    try:
        # 1. Create Metadata Entry
        upload_entry = DataUpload(
            user_id=user_id,
            filename=document_name,
            data_type='text',
            storage_location=f"pinecone://{pinecone_index_name}", # Store index name
            status='processing'
        )
        db.session.add(upload_entry)
        db.session.commit() # Commit to get the upload_id
        pinecone_namespace = f"upload-{upload_entry.id}" # Define namespace *after* getting ID

        # 2. Chunk the text
        # Adjust chunk_size and chunk_overlap as needed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text_content)
        if not chunks:
            raise ValueError("No text chunks generated from the input.")
        print(f"Split text '{document_name}' (Upload ID: {upload_entry.id}) into {len(chunks)} chunks.")

        # 3. Embed chunks
        model = get_embedding_model()
        print("Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True).tolist() # Convert to list
        print("Embeddings generated.")

        # 4. Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        vector_count = 0
        for i, chunk in enumerate(chunks):
            vector_id = f"vec-{upload_entry.id}-{i}" # Unique ID for each vector
            metadata = {
                'upload_id': upload_entry.id,
                'user_id': user_id,
                'document_name': document_name,
                'chunk_index': i,
                'text': chunk # Store the original text chunk in metadata
            }
            vectors_to_upsert.append((vector_id, embeddings[i], metadata))
            vector_count += 1

        # 5. Upsert vectors to Pinecone in batches
        batch_size = 100 # Pinecone recommendation
        print(f"Upserting {vector_count} vectors to Pinecone index '{pinecone_index_name}' namespace '{pinecone_namespace}'...")
        for i in range(0, len(vectors_to_upsert), batch_size):
             batch = vectors_to_upsert[i:i + batch_size]
             upsert_response = pinecone_index.upsert(vectors=batch, namespace=pinecone_namespace)
             print(f"Upserted batch {i//batch_size + 1}, response: {upsert_response}")

        # 6. Update Metadata Entry
        upload_entry.status = 'completed'
        upload_entry.vector_count = vector_count
        upload_entry.storage_location += f" (namespace: {pinecone_namespace})" # Add namespace info
        db.session.commit()
        print(f"Successfully processed text upload ID {upload_entry.id}")
        return True, f"Successfully processed text '{document_name}' ({vector_count} vectors created).", upload_entry.id

    except Exception as e:
        db.session.rollback()
        error_msg = f"Failed to process text upload '{document_name}': {e}"
        print(error_msg)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        if upload_entry and upload_entry.id:
             try:
                 entry = db.session.get(DataUpload, upload_entry.id)
                 if entry:
                     entry.status = 'failed'
                     entry.error_message = str(e)[:1000]
                     db.session.commit()
             except Exception as inner_e:
                 print(f"Additionally failed to update upload status to failed: {inner_e}")
                 db.session.rollback()

        # Attempt to delete potentially partially uploaded vectors in the namespace
        if pinecone_namespace and pinecone_index:
            try:
                 print(f"Attempting to delete potentially incomplete namespace '{pinecone_namespace}'...")
                 pinecone_index.delete(namespace=pinecone_namespace)
                 print(f"Deleted namespace '{pinecone_namespace}'.")
            except Exception as delete_e:
                 print(f"Failed to delete Pinecone namespace '{pinecone_namespace}' after error: {delete_e}")

        return False, error_msg, None