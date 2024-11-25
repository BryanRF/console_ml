from flask import Flask, render_template, request, redirect, url_for
import os
from main import main as run_training

app = Flask(__name__)

UPLOAD_FOLDER = './imagen'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return "No se encontró el archivo", 400
        
        dataset = request.files['dataset']
        if dataset.filename == '':
            return "Selecciona un archivo", 400

        # Guardar el dataset
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        dataset.save(dataset_path)
        return redirect(url_for('train', dataset_name=dataset.filename))
    return render_template('upload.html')

@app.route('/train/<dataset_name>')
def train(dataset_name):
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    results = run_training(dataset_path, dataset_name)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
