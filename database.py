import sqlite3

# Crear las tablas necesarias
def setup_database():
    conn = sqlite3.connect('modelos.db')
    cursor = conn.cursor()
    # Tabla para los datasets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            ruta_excel TEXT,
            ruta_pdf TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Tabla para los entrenamientos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entrenamientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_algoritmo TEXT,
            nombre_dataset TEXT,
            ruta_modelo TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Guardar un nuevo dataset
def save_dataset_to_db(nombre, ruta_excel, ruta_pdf):
    conn = sqlite3.connect('modelos.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO datasets (nombre, ruta_excel, ruta_pdf)
        VALUES (?, ?, ?)
    ''', (nombre, ruta_excel, ruta_pdf))
    conn.commit()
    conn.close()

# Guardar un nuevo entrenamiento
def save_training_to_db(nombre_algoritmo, nombre_dataset, ruta_modelo):
    conn = sqlite3.connect('modelos.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO entrenamientos (nombre_algoritmo, nombre_dataset, ruta_modelo)
        VALUES (?, ?, ?)
    ''', (nombre_algoritmo, nombre_dataset, ruta_modelo))
    conn.commit()
    conn.close()

# Recuperar el último dataset
def get_last_dataset():
    conn = sqlite3.connect('modelos.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM datasets
        ORDER BY fecha DESC
        LIMIT 1
    ''')
    dataset = cursor.fetchone()  # Solo un resultado
    conn.close()
    return dataset

# Recuperar entrenamientos asociados al último dataset
def get_trainings_for_last_dataset():
    conn = sqlite3.connect('modelos.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM entrenamientos
        WHERE nombre_dataset = (
            SELECT nombre FROM datasets
            ORDER BY fecha DESC
            LIMIT 1
        )
        ORDER BY fecha DESC
    ''')
    trainings = cursor.fetchall()  # Todos los entrenamientos relacionados
    conn.close()
    return trainings
