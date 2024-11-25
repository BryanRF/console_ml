import customtkinter as ctk
from templates.training_tab import create_training_tab
from templates.classification_tab import create_classification_tab

# Configuración de la ventana principal
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Sistema de Clasificación de Imágenes")
app.geometry("800x700")

# Crear Tabs
tabview = ctk.CTkTabview(app)
tabview.pack(fill="both", expand=True, padx=20, pady=20)

# Agregar pestañas
create_training_tab(tabview)
create_classification_tab(tabview)

# Ejecutar la ventana principal
app.mainloop()
