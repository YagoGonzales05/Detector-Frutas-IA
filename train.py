import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Configuración de rutas y parámetros
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# 2. Generadores de datos con aumento de datos básico
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2, # 80% para entrenamiento, 20% para validación
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

print("Preparando generadores de datos...")

# Dataset de entrenamiento
trainDataset = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Cambiado a 'categorical' para 3 o más clases
    subset='training'
)

# Dataset de validación
valDataset = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Cargar MobileNetV2 preentrenado (sin la capa de clasificación final)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Congelar las capas base

# 4. Añadir capas personalizadas para las 3 clases
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
# Salida de 3 neuronas con softmax porque ahora son 3 clases
output_layer = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# 5. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("¡Iniciando entrenamiento...")

# 6. Entrenar el modelo
history = model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=EPOCHS
)

# 7. Guardar el modelo resultante
model.save('modelo_frutas_3clases.h5')
print("¡Modelo entrenado y guardado con éxito como 'modelo_frutas_3clases.h5'!")