from flask_wtf import FlaskForm
from wtforms import StringField, SelectMultipleField, FileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

class TrainForm(FlaskForm):
    dataset_path = StringField('Ruta del Dataset', validators=[DataRequired()])
    dataset_name = StringField('Nombre del Dataset', validators=[DataRequired()])
    selected_algorithms = SelectMultipleField(
        'Selecciona Algoritmos',
        choices=[('SVM', 'SVM'), ('Naive Bayes', 'Naive Bayes'), 
                 ('Decision Tree', 'Decision Tree'), ('Logistic Regression', 'Logistic Regression'),
                 ('Neural Network', 'Neural Network')],
        validators=[DataRequired()]
    )

class ClassifyForm(FlaskForm):
    image = FileField('Seleccionar Imagen', validators=[FileAllowed(['jpg', 'png'], 'Solo imágenes permitidas')])
    model = StringField('Modelo Entrenado', validators=[DataRequired()])
