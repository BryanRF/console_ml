
#training_tab.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from main import main
import os
import customtkinter as ctk
from tkinter import messagebox
from database import get_last_dataset
import os

# Función para seleccionar una carpeta
def select_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_widget.delete(0, ctk.END)
        entry_widget.insert(0, folder_path)
        
def get_selected_algorithms(check_vars, algo_names):
    selected = [algo for algo, var in zip(algo_names, check_vars) if var.get()]
    if not selected:
        raise ValueError("Debes seleccionar al menos un algoritmo.")
    return selected
# Función para entrenar modelos
def start_training(
    entry_path, entry_name, progress_label, progress_bar, check_vars, algo_names, frame_to_hide, update_last_dataset_name
):
    dataset_path = entry_path.get()
    dataset_name = entry_name.get()

    if not dataset_path or not dataset_name:
        messagebox.showerror("Error", "Por favor, completa todos los campos.")
        return

    try:
        selected_algorithms = get_selected_algorithms(check_vars, algo_names)
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        return

    # Ocultar los elementos debajo del separador
    frame_to_hide.pack_forget()

    progress_label.configure(text="Entrenando modelos...")
    progress_bar.pack(pady=10)
    progress_bar.start()

    # Crear un hilo para no bloquear la interfaz
    thread = threading.Thread(
        target=run_training,
        args=(
            dataset_path,
            dataset_name,
            progress_label,
            progress_bar,
            frame_to_hide,
            selected_algorithms,
            update_last_dataset_name,
        ),
    )
    thread.start()

def run_training(
    dataset_path, dataset_name, progress_label, progress_bar, frame_to_hide, selected_algorithms, update_last_dataset_name
):
    try:
        main(dataset_path, dataset_name, selected_algorithms=selected_algorithms)  # Ejecuta el entrenamiento
        messagebox.showinfo("Éxito", "¡Entrenamiento completado!")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error durante el entrenamiento: {e}")
    finally:
        # Mostrar los elementos ocultos
        frame_to_hide.pack(pady=10)
        progress_bar.pack_forget()
        progress_label.configure(text="Listo")

        # Actualizar el último dataset cargado
        last_dataset = get_last_dataset()
        if last_dataset:
            progress_label.configure(text=f"Dataset Cargado: {last_dataset[1]}")
        else:
            progress_label.configure(text="No hay datasets registrados.")
            
        update_last_dataset_name()

# Función para abrir Excel del último dataset
def open_excel():
    last_dataset = get_last_dataset()
    if last_dataset:
        excel_path = last_dataset[2]  # Ruta del archivo Excel
        if os.path.exists(excel_path):
            os.startfile(excel_path)
        else:
            messagebox.showerror("Error", "El archivo Excel no existe.")
    else:
        messagebox.showerror("Error", "No hay datasets registrados.")

# Función para abrir PDF del último dataset
def open_pdf():
    last_dataset = get_last_dataset()
    print (last_dataset[3])
    if last_dataset:
        pdf_path = last_dataset[3]  # Ruta del archivo PDF
        if os.path.exists(pdf_path):
            os.startfile(pdf_path)
        else:
            messagebox.showerror("Error", "El archivo PDF no existe.")
    else:
        messagebox.showerror("Error", "No hay datasets registrados.")

# Contenido de la pestaña Entrenamiento
def create_training_tab(tabview):
    tab_training = tabview.add("Entrenamiento")

    # Sección 1: Cargar el dataset
    ctk.CTkLabel(tab_training, text="Carga del Dataset", font=("Arial", 16, "bold")).pack(pady=10)
    frame_path = ctk.CTkFrame(tab_training)
    frame_path.pack(pady=5)
    entry_path_training = ctk.CTkEntry(frame_path, width=400, placeholder_text="Selecciona la carpeta del dataset")
    entry_path_training.pack(side="left", padx=10)
    btn_browse_training = ctk.CTkButton(
        frame_path, text="Seleccionar", command=lambda: select_folder(entry_path_training)
    )
    btn_browse_training.pack(side="left")

    ctk.CTkLabel(tab_training, text="Nombre del Dataset:", font=("Arial", 14)).pack(pady=10)
    entry_name_training = ctk.CTkEntry(tab_training, width=400, placeholder_text="Ingresa el nombre del dataset")
    entry_name_training.pack(pady=5)

    # Lista de algoritmos con checkboxes
    algo_names = ['SVM', 'Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Neural Network']
    check_vars = [ctk.IntVar() for _ in algo_names]

    ctk.CTkLabel(tab_training, text="Selecciona Algoritmos:", font=("Arial", 14)).pack(pady=10)
    frame_algorithms = ctk.CTkFrame(tab_training)
    frame_algorithms.pack(pady=5)

    # Colocar checkboxes horizontalmente
    for col, (name, var) in enumerate(zip(algo_names, check_vars)):
        ctk.CTkCheckBox(frame_algorithms, text=name, variable=var).grid(row=0, column=col, padx=10, pady=5)

    progress_label_training = ctk.CTkLabel(tab_training, text="", font=("Arial", 12))
    progress_label_training.pack(pady=5)
    progress_bar_training = ctk.CTkProgressBar(tab_training, orientation="horizontal", mode="indeterminate", width=400)
    progress_bar_training.pack_forget()

    btn_train = ctk.CTkButton(
        tab_training,
        text="Entrenar Modelos",
        command=lambda: start_training(
            entry_path_training,
            entry_name_training,
            progress_label_training,
            progress_bar_training,
            check_vars,
            algo_names,
            frame_reports,
            update_last_dataset_name,
        ),
    )
    btn_train.pack(pady=5)
    separator = ctk.CTkFrame(tab_training, height=2, width=600, fg_color="gray")
    separator.pack(pady=10)

    # Mostrar el nombre del último dataset cargado
    last_dataset_name_label = ctk.CTkLabel(tab_training, text="Cargando...", font=("Arial", 14))
    last_dataset_name_label.pack(pady=5)

    # Contenedor para centrar los botones (frame para ocultar)
    frame_reports = ctk.CTkFrame(tab_training)
    frame_reports.pack(pady=10)

    # Botón para abrir Excel (verde)
    btn_open_excel = ctk.CTkButton(
        frame_reports, text="Abrir Excel", command=open_excel, width=150, fg_color="green", text_color="white"
    )
    btn_open_excel.grid(row=0, column=0, padx=20, pady=10)

    # Botón para abrir PDF (rojo)
    btn_open_pdf = ctk.CTkButton(
        frame_reports, text="Abrir PDF", command=open_pdf, width=150, fg_color="red", text_color="white"
    )
    btn_open_pdf.grid(row=0, column=1, padx=20, pady=10)

    # Función para actualizar el nombre del último dataset
    def update_last_dataset_name():
        last_dataset = get_last_dataset()
        if last_dataset:
            last_dataset_name_label.configure(text=f"Dataset Cargado: {last_dataset[1]}")
        else:
            last_dataset_name_label.configure(text="No hay datasets registrados.")

    # Llamar a la función al cargar la interfaz
    update_last_dataset_name()

    return update_last_dataset_name
