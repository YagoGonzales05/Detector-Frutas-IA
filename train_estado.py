
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# CONFIGURACION
DATASET_DIR = "dataset"
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 8
EPOCHS = 8

# AUMENTO DE DATOS PARA ENTRENAMIENTO
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

# CARGAR IMAGENES
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# PESOS PARA COMPENSAR EL DESBALANCE
counts = np.bincount(train_generator.classes)
total = sum(counts)

class_weights = {
    i: total / (len(counts) * count)
    for i, count in enumerate(counts)
}

print("\nClases detectadas:")
print(train_generator.class_indices)
print("\nPesos por clase:")
print(class_weights)

# MODELO MOBILENETV2
base_model = MobileNetV2(
    input_shape=(100, 100, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# CONTROL DEL ENTRENAMIENTO
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    )
]

# ENTRENAR
print("\nIniciando entrenamiento del detector de estado...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# EVALUAR
loss, accuracy = model.evaluate(validation_generator)

print("\n========== RESULTADOS ==========")
print(f"Exactitud de validacion: {accuracy * 100:.2f}%")
print(f"Perdida: {loss:.5f}")

# GUARDAR MODELO Y CLASES
model.save("modelo_estado.keras")

with open("clases_estado.json", "w", encoding="utf-8") as file:
    json.dump(
        train_generator.class_indices,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nModelo guardado correctamente.")
print("Archivos: modelo_estado.keras y clases_estado.json")
