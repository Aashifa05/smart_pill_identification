"""WORKING MODEL - Efficient TensorFlow approach"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import json
import numpy as np

print("\n" + "=" * 80)
print("TRAINING WORKING PILL CLASSIFIER MODEL")
print("=" * 80 + "\n")

TRAIN_DIR = 'media/pilldata/train'
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 40

CLASS_NAMES = [
    'Amoxicillin 500 MG', 'Atomoxetine 25 MG', 'Calcitriol 0.00025 MG',
    'Oseltamivir 45 MG', 'Ramipril 5 MG', 'apixaban 2.5 MG',
    'aprepitant 80 MG', 'benzonatate 100 MG', 'carvedilol 3.125 MG',
    'celecoxib 200 MG', 'duloxetine 30 MG', 'eltrombopag 25 MG',
    'montelukast 10 MG', 'mycophenolate mofetil 250 MG',
    'pantoprazole 40 MG', 'pitavastatin 1 MG', 'prasugrel 10 MG',
    'saxagliptin 5 MG', 'sitagliptin 50 MG', 'tadalafil 5 MG'
]

print("Step 1: Loading images...")

def get_class(filename):
    for cls in CLASS_NAMES:
        if cls.lower() in filename.lower():
            return cls
    return None

all_files = []
for f in os.listdir(TRAIN_DIR):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        cls = get_class(f)
        if cls:
            all_files.append((os.path.join(TRAIN_DIR, f), CLASS_NAMES.index(cls)))

print(f"Found {len(all_files)} images")

np.random.seed(42)
np.random.shuffle(all_files)
split = int(0.8 * len(all_files))
train_files = all_files[:split]
val_files = all_files[split:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}\n")

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices(([p for p,l in train_files], [l for p,l in train_files]))
train_ds = train_ds.map(load_img, tf.data.AUTOTUNE)
train_ds = train_ds.map(augment, tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(([p for p,l in val_files], [l for p,l in val_files]))
val_ds = val_ds.map(load_img, tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Step 2: Building model...")

base = keras.applications.MobileNetV3Large(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

for layer in base.layers[:-40]:
    layer.trainable = False

model = keras.Sequential([
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Parameters: {model.count_params():,}\n")
print("Step 3: Training...\n")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ],
    verbose=1
)

print("\nStep 4: Saving...")

model.save('media/pilldata/model_working.keras')

final_val_acc = history.history['val_accuracy'][-1]
final_train_acc = history.history['accuracy'][-1]

metadata = {
    'class_names': CLASS_NAMES,
    'image_size': IMG_SIZE,
    'training_samples': len(train_files),
    'validation_samples': len(val_files),
    'total_params': int(model.count_params()),
    'final_train_accuracy': float(final_train_acc),
    'final_val_accuracy': float(final_val_acc),
    'epochs_trained': len(history.history['loss']),
}

with open('media/pilldata/model_working_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE")
print("=" * 80)
print(f"\nValidation Accuracy: {final_val_acc:.1%}")
print(f"Training Accuracy: {final_train_acc:.1%}")
print(f"Model: media/pilldata/model_working.keras\n")
