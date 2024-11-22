import sqlite3

# Crear o conectar la base de datos
def initialize_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS algorithms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Insertar dataset
def insert_dataset(name, path):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO datasets (name, path) VALUES (?, ?)', (name, path))
    conn.commit()
    conn.close()

# Insertar algoritmo
def insert_algorithm(name, model_path):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO algorithms (name, model_path) VALUES (?, ?)', (name, model_path))
    conn.commit()
    conn.close()

# Obtener todos los datasets
def get_datasets():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, path FROM datasets')
    datasets = cursor.fetchall()
    conn.close()
    return datasets

# Obtener todos los algoritmos
def get_algorithms():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, model_path FROM algorithms')
    algorithms = cursor.fetchall()
    conn.close()
    return algorithms
