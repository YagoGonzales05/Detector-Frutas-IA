
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "modelo_estado.keras"
CLASSES_PATH = "clases_estado.json"

# Comprobar archivos
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontro modelo_estado.keras")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError("No se encontro clases_estado.json")

# Cargar modelo
print("Cargando detector de estado...")
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar clases
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_indices = json.load(file)

idx_to_class = {
    index: name for name, index in class_indices.items()
}

def detectar_estado(ruta_imagen):

    if not os.path.exists(ruta_imagen):
        print("No se encontro la imagen:", ruta_imagen)
        return

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print("No se pudo abrir la imagen.")
        return

    # Convertir BGR a RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Redimensionar
    imagen = cv2.resize(imagen, (100, 100))

    # Preparar imagen
    imagen = np.array(imagen, dtype=np.float32)
    imagen = preprocess_input(imagen)
    imagen = np.expand_dims(imagen, axis=0)

    # Prediccion
    predicciones = model.predict(imagen, verbose=0)[0]

    indice = int(np.argmax(predicciones))
    estado = idx_to_class[indice]
    confianza = float(predicciones[indice]) * 100

    print("\n========== ESTADO DE LA FRUTA ==========")
    print(f"Estado visual: {estado}")
    print(f"Puntuacion del modelo: {confianza:.2f}%")
    print("========================================")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ruta = " ".join(sys.argv[1:])
    else:
        ruta = "imagen_prueba.jpg"

    detectar_estado(ruta)
