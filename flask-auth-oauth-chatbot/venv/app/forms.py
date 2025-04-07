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


