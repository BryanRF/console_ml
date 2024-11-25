
# Configuración del Entorno Virtual

Este proyecto utiliza un entorno virtual para manejar las dependencias. A continuación, se explican los pasos necesarios para configurarlo.

## Requisitos previos

Asegúrate de tener instalado:
- Python 3.8.10 o superior
- `pip` (viene incluido con Python)

## Pasos de configuración

1. **Instalar `virtualenv` (si no lo tienes ya instalado):**
   ```bash
   pip install virtualenv
   ```

2. **Crear el entorno virtual llamado `venv`:**
   ```bash
   virtualenv venv
   ```

3. **Activar el entorno virtual:**
   - En **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - En **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias del proyecto:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar el entorno virtual:**
   Una vez activado el entorno, puedes ejecutar tu proyecto normalmente dentro de este entorno.

6. **Salir del entorno virtual (opcional):**
   Cuando termines, puedes desactivar el entorno virtual con:
   ```bash
   deactivate
   ```
7. **Ejecutar proyecto:**
   Cuando termines, puedes desactivar el entorno virtual con:
   ```bash
   python run.py
   ```

## Notas
- Cada vez que trabajes en este proyecto, recuerda activar el entorno virtual antes de ejecutar cualquier comando.
- Si necesitas congelar nuevas dependencias instaladas, usa:
  ```bash
  pip freeze > requirements.txt
  ```
