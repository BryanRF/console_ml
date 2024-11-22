import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from main import main, classify_image

# Configurar el tema
ctk.set_appearance_mode("Dark")  # Opciones: "Dark", "Light", "System"
ctk.set_default_color_theme("blue")  # Colores disponibles: "blue", "green", "dark-blue"

# Función para seleccionar una carpeta
def select_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_widget.delete(0, ctk.END)
        entry_widget.insert(0, folder_path)

# Función para entrenar modelos
def start_training(entry_path, entry_name, progress_label, progress_bar):
    dataset_path = entry_path.get()
    dataset_name = entry_name.get()

    if not dataset_path or not dataset_name:
        messagebox.showerror("Error", "Por favor, completa todos los campos.")
        return

    progress_label.configure(text="Entrenando modelos...")
    progress_bar.pack(pady=10)  # Mostrar la barra de progreso
    progress_bar.start()

    # Crear un hilo para no bloquear la interfaz
    thread = threading.Thread(target=run_training, args=(dataset_path, dataset_name, progress_label, progress_bar))
    thread.start()

def run_training(dataset_path, dataset_name, progress_label, progress_bar):
    try:
        main(dataset_path, dataset_name)  # Llama la función `main` del archivo `main.py`
        messagebox.showinfo("Éxito", "¡Entrenamiento completado!")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error durante el entrenamiento: {e}")
    finally:
        progress_bar.pack_forget()  # Ocultar la barra de progreso
        progress_label.configure(text="Listo")

# Función para clasificar una imagen
def classify(entry_image, entry_model):
    image_path = entry_image.get()
    model_path = entry_model.get()

    if not image_path or not model_path:
        messagebox.showerror("Error", "Por favor, selecciona una imagen y un modelo.")
        return

    try:
        prediction = classify_image(image_path, model_path)  # Llama a `classify_image` de `main.py`
        messagebox.showinfo("Resultado", f"La clase predicha es: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error: {e}")

# Crear la ventana principal
app = ctk.CTk()
app.title("Sistema de Clasificación de Imágenes")
app.geometry("700x500")

# Crear Tabs usando CTkTabview
tabview = ctk.CTkTabview(app)
tabview.pack(fill="both", expand=True, padx=20, pady=20)

# Crear Pestañas
tab_training = tabview.add("Entrenamiento")
tab_classification = tabview.add("Clasificación")

# Contenido de la pestaña Entrenamiento
ctk.CTkLabel(tab_training, text="Ruta del Dataset:", font=("Arial", 14)).pack(pady=10)
frame_path = ctk.CTkFrame(tab_training)
frame_path.pack(pady=5)
entry_path_training = ctk.CTkEntry(frame_path, width=400, placeholder_text="Selecciona la carpeta del dataset")
entry_path_training.pack(side="left", padx=10)
btn_browse_training = ctk.CTkButton(frame_path, text="Seleccionar", command=lambda: select_folder(entry_path_training))
btn_browse_training.pack(side="left")

ctk.CTkLabel(tab_training, text="Nombre del Dataset:", font=("Arial", 14)).pack(pady=10)
entry_name_training = ctk.CTkEntry(tab_training, width=400, placeholder_text="Ingresa el nombre del dataset")
entry_name_training.pack(pady=5)

btn_train = ctk.CTkButton(tab_training, text="Entrenar Modelos", command=lambda: start_training(
    entry_path_training, entry_name_training, progress_label_training, progress_bar_training))
btn_train.pack(pady=20)

progress_label_training = ctk.CTkLabel(tab_training, text="", font=("Arial", 12))
progress_label_training.pack(pady=5)
progress_bar_training = ctk.CTkProgressBar(tab_training, orientation="horizontal", mode="indeterminate", width=400)
progress_bar_training.pack_forget() 

# Contenido de la pestaña Clasificación
ctk.CTkLabel(tab_classification, text="Ruta de la Imagen:", font=("Arial", 14)).pack(pady=10)
frame_image = ctk.CTkFrame(tab_classification)
frame_image.pack(pady=5)
entry_image = ctk.CTkEntry(frame_image, width=400, placeholder_text="Selecciona la imagen a clasificar")
entry_image.pack(side="left", padx=10)
btn_browse_image = ctk.CTkButton(frame_image, text="Seleccionar", command=lambda: select_folder(entry_image))
btn_browse_image.pack(side="left")

ctk.CTkLabel(tab_classification, text="Ruta del Modelo:", font=("Arial", 14)).pack(pady=10)
frame_model = ctk.CTkFrame(tab_classification)
frame_model.pack(pady=5)
entry_model = ctk.CTkEntry(frame_model, width=400, placeholder_text="Selecciona el modelo preentrenado")
entry_model.pack(side="left", padx=10)
btn_browse_model = ctk.CTkButton(frame_model, text="Seleccionar", command=lambda: select_folder(entry_model))
btn_browse_model.pack(side="left")

btn_classify = ctk.CTkButton(tab_classification, text="Clasificar Imagen", command=lambda: classify(entry_image, entry_model))
btn_classify.pack(pady=20)

# Ejecutar la ventana principal
app.mainloop()
