import os
from flask import Blueprint, render_template, request, redirect, url_for, send_file,jsonify
from werkzeug.utils import secure_filename
from app.forms import TrainForm, ClassifyForm
from app.utils import *  # Ahora desde utils.py
import zipfile
import rarfile

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/train', methods=['GET', 'POST'])
def train():
    form = TrainForm()

    if request.method == 'GET':
        # Renderizar la página de entrenamiento (para cargar el formulario)
        return render_template('train.html', form=form)

    if request.method == 'POST':
        try:
            dataset_file = request.files.get('dataset_file')
            dataset_name = request.form.get('dataset_name')
            selected_algorithms = request.form.getlist('selected_algorithms')

            # Validar si se subió un archivo
            if not dataset_file or dataset_file.filename == '':
                return jsonify({'error': 'Por favor selecciona un archivo comprimido.'}), 400

            # Guardar el archivo comprimido
            upload_folder = 'app/static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, secure_filename(dataset_file.filename))
            dataset_file.save(file_path)

            # Extraer archivo comprimido
            extract_path = os.path.join(upload_folder, 'extracted', dataset_name)
            os.makedirs(extract_path, exist_ok=True)
            try:
                if file_path.endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                elif file_path.endswith('.rar'):
                    with rarfile.RarFile(file_path, 'r') as rar_ref:
                        rar_ref.extractall(extract_path)
                else:
                    return jsonify({'error': 'Formato no soportado. Usa .zip o .rar.'}), 400
            except Exception as e:
                return jsonify({'error': f'Error al descomprimir el archivo: {str(e)}'}), 500

            # Llamar a la función de entrenamiento
            try:
                main(extract_path, dataset_name, selected_algorithms=selected_algorithms)
                return jsonify({'message': '¡Entrenamiento completado exitosamente!'})
            except Exception as e:
                return jsonify({'error': f'Error durante el entrenamiento: {str(e)}'}), 500
        except Exception as e:
            return jsonify({'error': f'Ocurrió un error inesperado: {str(e)}'}), 500

@main_bp.route('/classify', methods=['GET', 'POST'])
def classify():
    form = ClassifyForm()

    if request.method == 'GET':
        # Renderizar la página de clasificación
        return render_template('classify.html', form=form)

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            image = request.files.get('image')
            model_path = request.form.get('model')

            # Validar los datos recibidos
            if not image or not model_path:
                return jsonify({'error': 'Debe proporcionar una imagen y un modelo.'}), 400

            # Guardar la imagen temporalmente
            upload_folder = 'app/static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filename = secure_filename(image.filename)
            image_path = os.path.join(upload_folder, filename)
            image.save(image_path)

            # Llamar a la función de clasificación
            prediction = classify_image(image_path, model_path)
            return jsonify({'message': 'Clasificación completada', 'prediction': prediction})
        except Exception as e:
            return jsonify({'error': f'Error al clasificar la imagen: {str(e)}'}), 500
        
        
        
@main_bp.route('/get_models', methods=['GET'])
def get_models():
    try:
        # Recuperar entrenamientos asociados al último dataset
        trainings = get_trainings_for_last_dataset()

        # Procesar los entrenamientos en una lista de opciones para el dropdown
        model_options = [{'name': training[1], 'path': training[3]} for training in trainings]

        if not model_options:
            return jsonify({'error': 'No se encontraron modelos disponibles'}), 404

        return jsonify({'models': model_options})
    except Exception as e:
        return jsonify({'error': f'Error al obtener los modelos: {str(e)}'}), 500

@main_bp.route('/get_last_dataset', methods=['GET'])
def get_last_dataset_info():
    try:
        # Recuperar el último dataset desde la base de datos
        dataset = get_last_dataset()
        if not dataset:
            return jsonify({'error': 'No hay datasets registrados'}), 404

        # Crear respuesta JSON con información del dataset
        dataset_info = {
            'name': dataset[1],
            'excel_path': dataset[2],
            'pdf_path': dataset[3]
        }
        return jsonify(dataset_info)
    except Exception as e:
        return jsonify({'error': f'Error al obtener el dataset: {str(e)}'}), 500


@main_bp.route('/download/<file_type>/<path:file_path>', methods=['GET'])
def download_file(file_type, file_path):
    try:
        # Usar ruta absoluta correcta
        absolute_path = os.path.abspath(file_path)

        # Validar que el archivo existe
        if not os.path.exists(absolute_path):
            return jsonify({"error": f"Archivo {file_type} no encontrado: {absolute_path}"}), 404

        # Enviar el archivo al cliente
        return send_file(absolute_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error al descargar el archivo: {str(e)}"}), 500


