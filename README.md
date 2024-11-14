# Proyecto de Clasificación de Imágenes con Modelos de Machine Learning

Este proyecto entrena varios modelos de machine learning para la clasificación de imágenes, permitiendo evaluar cada modelo y generar un reporte con los resultados. Además, permite cargar una imagen y predecir su clase usando el mejor modelo entrenado.

## 1. Instalación

### Requisitos previos
Asegúrate de tener Python 3.6 o superior y `pip` instalado en tu sistema.

### Instalación de Dependencias

1. Clona el repositorio o descarga los archivos del proyecto.
2. Navega al directorio del proyecto en la terminal.
3. Ejecuta el siguiente comando para instalar todas las dependencias del proyecto:

   ```bash
   pip install -r requirements.txt
Nota: El archivo requirements.txt debe contener todas las bibliotecas necesarias, como numpy, pandas, scikit-learn, Pillow, openpyxl, y cualquier otra dependencia utilizada en el proyecto.

2. Ejecución
Entrenamiento y Generación de Reporte
Para entrenar los modelos y generar un reporte de rendimiento en Excel, ejecuta el archivo principal de la siguiente manera:

bash
Copiar código
python main.py "ruta_del_dataset" "nombre_del_dataset" "source"
Ejemplo
bash
Copiar código
python main.py "C:\\Users\\rfrey\\Documents\\console_ml\\dataset" "Dataset_de_Estrias" "local"
Este comando:

Entrena los modelos especificados en el código (SVM, Naive Bayes, Decision Tree, Logistic Regression, y Neural Network).
Genera un archivo de reporte en Excel con el rendimiento de cada modelo, resaltando el mejor modelo basado en Accuracy.
Guarda cada modelo entrenado en una carpeta entrenamiento bajo el nombre del dataset.
Clasificación de Nueva Imagen
Para clasificar una nueva imagen usando el mejor modelo entrenado, puedes llamar a la función classify_image y pasar la ruta de la imagen y el modelo:

python
Copiar código
image_class = classify_image("ruta_de_la_imagen", "ruta_del_mejor_modelo.pkl")
print(f"🧠 La clase de la imagen es: {image_class}")
Ejemplo
python
Copiar código
image_class = classify_image("C:\\Users\\rfrey\\Documents\\console_ml\\imagen\\e204236c65.JPG", "entrenamiento/Dataset_de_Estrias/SVM.pkl")
print(f"🧠 La clase de la imagen es: {image_class}")
Esta función cargará el modelo especificado, procesará la imagen y devolverá el nombre de la clase predicha.

3. Explicación del Proyecto
Este proyecto está diseñado para automatizar el proceso de entrenamiento, evaluación y uso de modelos de clasificación de imágenes. Los componentes principales son:

Entrenamiento de Modelos: Utiliza varios modelos de machine learning para el reconocimiento de imágenes, como SVM, Naive Bayes, Árbol de Decisión, Regresión Logística y Redes Neuronales. Cada modelo se entrena con los datos proporcionados, y se guarda el modelo entrenado en la carpeta entrenamiento.

Generación de Reporte: Tras el entrenamiento, se genera un archivo de reporte en formato Excel que contiene métricas de precisión, sensibilidad (recall), puntaje F1, AUC, uso de CPU y tiempo de ejecución de cada modelo. La mejor fila del reporte se resalta para destacar el modelo con mejor rendimiento.

Clasificación de Imágenes Nuevas: Puedes cargar una nueva imagen para que el sistema clasifique su clase utilizando el mejor modelo guardado. Esto facilita el uso del sistema para clasificar imágenes después del entrenamiento inicial.

Organización de Archivos y Carpetas
entrenamiento/: Contiene los modelos entrenados para cada dataset, organizados en carpetas.
resultados/: Contiene los reportes de rendimiento generados en Excel, cada uno con la fecha y hora de creación.
main.py: Archivo principal que contiene el código para el entrenamiento y clasificación.
requirements.txt: Archivo con todas las dependencias necesarias para el proyecto.
Este proyecto es ideal para la implementación de un pipeline de clasificación de imágenes, donde se pueden comparar y almacenar modelos, permitiendo además la clasificación rápida de nuevas imágenes.

¡Gracias por usar este proyecto de clasificación de imágenes con machine learning! 🚀