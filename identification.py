
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Archivos del modelo
MODEL_PATH = "modelo_identificacion.keras"
CLASSES_PATH = "clases_identificacion.json"

# Comprobar que existan los archivos
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo entrenado.")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError("No se encontró el archivo de clases.")

# Cargar modelo
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar nombres de clases
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_indices = json.load(file)

# Convertir índices a nombres
idx_to_class = {
    index: name for name, index in class_indices.items()
}

def identificar_fruta(ruta_imagen):

    if not os.path.exists(ruta_imagen):
        print("No se encontró la imagen:", ruta_imagen)
        return

    # Leer imagen
    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print("No se pudo abrir la imagen.")
        return

    # OpenCV utiliza BGR, MobileNetV2 necesita RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Redimensionar a 100 x 100
    imagen = cv2.resize(imagen, (100, 100))

    # Preparar imagen para MobileNetV2
    imagen = np.array(imagen, dtype=np.float32)
    imagen = preprocess_input(imagen)
    imagen = np.expand_dims(imagen, axis=0)

    # Realizar predicción
    predicciones = model.predict(imagen, verbose=0)[0]

    # Obtener las tres clases con mayor puntuación
    mejores = np.argsort(predicciones)[::-1][:3]

    print("\n========== RESULTADO ==========")

    for posicion, indice in enumerate(mejores, start=1):

        nombre = idx_to_class.get(int(indice), "Desconocido")
        porcentaje = float(predicciones[indice]) * 100

        print(f"{posicion}. {nombre}: {porcentaje:.2f}%")

    print("===============================")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ruta = " ".join(sys.argv[1:])
    else:
        ruta = "imagen_prueba.jpg"

    identificar_fruta(ruta)
