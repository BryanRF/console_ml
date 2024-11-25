import customtkinter as ctk
from tkinter import filedialog, messagebox
from database import get_trainings_for_last_dataset
from main import classify_image
import os
# Función para seleccionar una imagen
def select_file(entry_widget):
    file_path = filedialog.askopenfilename(
        title="Seleccionar Imagen",
        filetypes=[("Archivos de Imagen", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if file_path:
        entry_widget.delete(0, ctk.END)
        entry_widget.insert(0, file_path)
        print(f"[DEBUG] Imagen seleccionada: {file_path}")  # Depuración

# Función para clasificar una imagen
def classify(entry_image, dropdown_model):
    image_path = entry_image.get()
    model_path = dropdown_model.get()  # Esto debe ser una ruta válida a un archivo .pkl

    print(f"[DEBUG] Ruta de la imagen seleccionada: {image_path}")  # Depuración
    print(f"[DEBUG] Ruta del modelo seleccionado: {model_path}")  # Depuración

    if not image_path or not os.path.exists(image_path):
        print(f"[ERROR] La imagen no existe o la ruta es incorrecta: {image_path}")  # Depuración
        messagebox.showerror("Error", "Selecciona una imagen válida.")
        return

    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] El modelo no existe o la ruta es incorrecta: {model_path}")  # Depuración
        messagebox.showerror("Error", "Selecciona un modelo válido.")
        return

    try:
        prediction = classify_image(image_path, model_path)
        print(f"[DEBUG] Predicción obtenida: {prediction}")  # Depuración
        messagebox.showinfo("Resultado", f"La clase predicha es: {prediction}")
    except Exception as e:
        print(f"[ERROR] Error al clasificar la imagen: {e}")  # Depuración
        messagebox.showerror("Error", f"Error al clasificar la imagen: {e}")

# Función para actualizar el Dropdown de modelos
def update_model_dropdown(dropdown):
    trainings = get_trainings_for_last_dataset()
    print(f"[DEBUG] Entrenamientos obtenidos: {trainings}")  # Depuración
    if trainings:
        # Almacenar solo las rutas reales de los modelos en el dropdown
        model_paths = [t[3] for t in trainings if os.path.exists(t[3])]
        print(f"[DEBUG] Rutas de modelos válidas: {model_paths}")  # Depuración
        dropdown.configure(values=model_paths, width=500)
        if model_paths:
            dropdown.set(model_paths[0])  # Seleccionar el modelo más reciente
        else:
            dropdown.set("No hay modelos disponibles")
    else:
        print("[DEBUG] No se encontraron entrenamientos.")  # Depuración
        dropdown.configure(values=["No hay modelos disponibles"], width=500)
        dropdown.set("No hay modelos disponibles")

# Función para refrescar el dropdown después de entrenar
def refresh_model_list(dropdown):
    print("[DEBUG] Refrescando lista de modelos...")  # Depuración
    update_model_dropdown(dropdown)
    messagebox.showinfo("Actualización", "¡Lista de modelos actualizada con éxito!")

# Contenido de la pestaña Clasificación
# Contenido de la pestaña Clasificación
def create_classification_tab(tabview):
    tab_classification = tabview.add("Clasificación")

    # Título de selección de imagen
    ctk.CTkLabel(tab_classification, text="Ruta de la Imagen:", font=("Arial", 14)).pack(pady=10)
    
    # Entrada para seleccionar imagen
    frame_image = ctk.CTkFrame(tab_classification)
    frame_image.pack(pady=5)
    entry_image = ctk.CTkEntry(frame_image, width=400, placeholder_text="Selecciona la imagen a clasificar")
    entry_image.pack(side="left", padx=10)
    btn_browse_image = ctk.CTkButton(frame_image, text="Seleccionar", command=lambda: select_file(entry_image))
    btn_browse_image.pack(side="left")

    # Dropdown y botón de refrescar modelos en la misma fila
    ctk.CTkLabel(tab_classification, text="Seleccionar Modelo Entrenado:", font=("Arial", 14)).pack(pady=10)
    frame_dropdown = ctk.CTkFrame(tab_classification)
    frame_dropdown.pack(pady=10)

    # Dropdown de modelos
    model_dropdown = ctk.CTkComboBox(frame_dropdown, values=["Cargando modelos..."], width=400)
    model_dropdown.pack(side="left", padx=10)

    # Botón para refrescar modelos
    btn_refresh_models = ctk.CTkButton(
        frame_dropdown, text="Refrescar Modelos", command=lambda: refresh_model_list(model_dropdown)
    )
    btn_refresh_models.pack(side="left")

    # Botón para clasificar imagen
    btn_classify = ctk.CTkButton(tab_classification, text="Clasificar Imagen", command=lambda: classify(entry_image, model_dropdown))
    btn_classify.pack(pady=20)

    # Inicializar el dropdown
    update_model_dropdown(model_dropdown)
