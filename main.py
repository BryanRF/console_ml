from app import create_app

# Crear la instancia de la aplicación Flask
app = create_app()

if __name__ == "__main__":
    # Ejecutar la aplicación Flask
    app.run(debug=True)
