import cv2
import numpy as np
import tensorflow as tf

# 1. Cargar el modelo
print("Cargando modelo...")
model = tf.keras.models.load_model('modelo_frutas_3clases.h5')

# 2. Definir las clases
clases = ['Leve', 'Podrido', 'Recién publicadas']

# 3. Abrir la cámara web
cap = cv2.VideoCapture(0)

print("¡Cámara lista! Presiona la letra 'q' o 'ESC' para salir.")

contador = 0
clase_nombre = "Analizando..."
confianza = 0.0
color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contador += 1

    if contador % 15 == 0:
        img_resized = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        prediccion = model.predict(img_array, verbose=0)
        clase_idx = np.argmax(prediccion[0])
        confianza = prediccion[0][clase_idx] * 100
        clase_nombre = clases[clase_idx]

        if clase_nombre == 'Recién publicadas':
            color = (0, 255, 0)      # Verde
        elif clase_nombre == 'Leve':
            color = (0, 165, 255)    # Naranja
        else:
            color = (0, 0, 255)      # Rojo

    texto = f"{clase_nombre} ({confianza:.1f}%)"
    cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow('Detector de Frescura de Frutas', frame)

    # Capturar la tecla presionada (waitKey(25) da mejor respuesta)
    key = cv2.waitKey(25) & 0xFF
    
    # Salir si presionan 'q' o la tecla 'ESC' (código 27)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()