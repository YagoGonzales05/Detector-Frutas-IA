
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CONFIGURACION
# ==========================================

IMAGE_SIZE = (100, 100)
BATCH_SIZE = 8
EPOCHS = 5

TRAIN_DIR = Path("fruits360/Training")
TEST_DIR = Path("fruits360/Test")

# ==========================================
# 2. VERIFICAR CARPETAS
# ==========================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"No se encuentra: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"No se encuentra: {TEST_DIR}")

# ==========================================
# 3. SELECCIONAR 10 VARIEDADES
# ==========================================

all_classes = sorted([
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
])

# Intentamos seleccionar frutas variadas
preferred = [
    "Apple Golden",
    "Banana",
    "Grape Blue",
    "Kiwi",
    "Mango",
    "Orange",
    "Pear",
    "Pineapple",
    "Strawberry",
    "Watermelon"
]

selected_classes = []

for prefix in preferred:
    matches = [
        name for name in all_classes
        if name.lower().startswith(prefix.lower())
        and name not in selected_classes
    ]

    if matches:
        selected_classes.append(matches[0])

# Si alguna fruta no existe, completamos con otras clases
for name in all_classes:
    if len(selected_classes) >= 10:
        break

    if name not in selected_classes:
        selected_classes.append(name)

selected_classes = selected_classes[:10]

if len(selected_classes) < 2:
    raise ValueError("No se encontraron suficientes clases.")

for name in selected_classes:
    if not (TEST_DIR / name).is_dir():
        raise FileNotFoundError(
            f"La clase '{name}' no existe en la carpeta Test."
        )

print("\nVARIEDADES SELECCIONADAS:")
for name in selected_classes:
    print("-", name)

print("\nTotal de clases:", len(selected_classes))

# ==========================================
# 4. PREPARAR IMAGENES
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

trainDataset = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    classes=selected_classes,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

valDataset = validation_datagen.flow_from_directory(
    str(TRAIN_DIR),
    classes=selected_classes,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# ==========================================
# 5. CREAR MOBILENETV2
# ==========================================

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

output_layer = Dense(
    len(selected_classes),
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output_layer
)

# ==========================================
# 6. COMPILAR MODELO
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 7. CONFIGURAR ENTRENAMIENTO
# ==========================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
]

# ==========================================
# 8. ENTRENAR
# ==========================================

history = model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 9. EVALUAR CON TEST
# ==========================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

testDataset = test_datagen.flow_from_directory(
    str(TEST_DIR),
    classes=selected_classes,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(testDataset)

print("\nRESULTADOS:")
print("Perdida:", test_loss)
print("Precision:", test_accuracy * 100, "%")

# ==========================================
# 10. GUARDAR MODELO Y CLASES
# ==========================================

model.save("modelo_identificacion.keras")

with open(
    "clases_identificacion.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        trainDataset.class_indices,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nENTRENAMIENTO FINALIZADO")
print("Modelo guardado correctamente.")
print("Archivo: modelo_identificacion.keras")
print("Clases: clases_identificacion.json")
