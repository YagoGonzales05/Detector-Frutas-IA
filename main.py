import cv2

print("1. Iniciando script...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir la cámara. ¿Está usada por otra aplicación?")
else:
    print("✅ ¡Cámara abierta con éxito! Presiona 'q' en la ventana de video para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo leer el fotograma.")
            break
        
        cv2.imshow("Prueba de Camara - Presiona Q", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente.")