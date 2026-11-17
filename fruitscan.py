
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# CONFIGURACION
# =====================================

IMG_SIZE = (100, 100)

# =====================================
# CARGAR MODELOS
# =====================================

print("Cargando FruitScan AI...")

modelo_identificacion = tf.keras.models.load_model(
    "modelo_identificacion.keras"
)

modelo_estado = tf.keras.models.load_model(
    "modelo_estado.keras"
)

# =====================================
# CARGAR CLASES
# =====================================

with open("clases_identificacion.json", "r", encoding="utf-8") as f:
    clases_identificacion = json.load(f)

with open("clases_estado.json", "r", encoding="utf-8") as f:
    clases_estado = json.load(f)

# Invertir diccionarios
id_a_fruta = {
    int(indice): nombre
    for nombre, indice in clases_identificacion.items()
}

id_a_estado = {
    int(indice): nombre
    for nombre, indice in clases_estado.items()
}

# =====================================
# PREPARAR IMAGEN
# =====================================

def preparar_imagen(ruta):

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise ValueError("No se pudo leer la imagen.")

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(imagen, IMG_SIZE)

    imagen = np.array(imagen, dtype=np.float32)
    imagen = preprocess_input(imagen)
    imagen = np.expand_dims(imagen, axis=0)

    return imagen

# =====================================
# ANALISIS COMPLETO
# =====================================


def analizar_fruta(ruta, mostrar=True):

    imagen = preparar_imagen(ruta)

    # Identificacion
    pred_fruta = modelo_identificacion.predict(imagen, verbose=0)[0]

    indice_fruta = int(np.argmax(pred_fruta))
    fruta = id_a_fruta[indice_fruta]
    confianza_fruta = float(pred_fruta[indice_fruta]) * 100

    # Estado visual
    pred_estado = modelo_estado.predict(imagen, verbose=0)[0]

    indice_estado = int(np.argmax(pred_estado))
    estado = id_a_estado[indice_estado]
    confianza_estado = float(pred_estado[indice_estado]) * 100

    resultado = {
        "fruta": fruta,
        "confianza_fruta": round(confianza_fruta, 2),
        "estado": estado,
        "confianza_estado": round(confianza_estado, 2)
    }

    if mostrar:
        print("\n========== FRUITSCAN AI ==========")
        print(f"Alimento identificado: {fruta}")
        print(f"Puntuacion: {confianza_fruta:.2f}%")
        print(f"Estado visual: {estado}")
        print(f"Puntuacion: {confianza_estado:.2f}%")
        print("==================================")

    return resultado

# =====================================
# EJECUCION
# =====================================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        ruta_imagen = " ".join(sys.argv[1:])
    else:
        ruta_imagen = "imagen_prueba.jpg"

    try:
        analizar_fruta(ruta_imagen)

    except Exception as error:
        print("\nError:", error)
