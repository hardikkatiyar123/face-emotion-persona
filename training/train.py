"""
Train an emotion-recognition CNN on FER-2013 and export it for the browser.

Dataset: https://www.kaggle.com/datasets/msambare/fer2013
Unzip it so you have:
    data/train/angry, data/train/disgust, ... (7 folders)
    data/test/angry,  data/test/disgust,  ...

Run:
    python train.py --data data --epochs 50

15% of train/ is held out for validation (early stopping, learning-rate schedule,
choosing the best epoch). test/ is used only once, at the end, so the reported
test accuracy is an honest estimate.

Outputs:
    emotion_model.h5          (Keras model, used by local_demo.py)
    ../web/model/model.json   (TensorFlow.js model, used by the website)
    ../web/model/*.bin
    ../web/model/labels.json
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 48
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")  # what image_dataset_from_directory reads

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data", help="folder containing train/ and test/")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--val-split", type=float, default=0.15, help="fraction of train/ held out for validation")
parser.add_argument("--out", default=os.path.join("..", "web", "model"))
args = parser.parse_args()

train_dir = os.path.join(args.data, "train")
test_dir = os.path.join(args.data, "test")

# ---------- 1. Load data ----------
common = dict(
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=args.batch,
    label_mode="categorical",
)
# Split train/ into train + validation (same seed = same split every run)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    train_dir, validation_split=args.val_split, subset="both", shuffle=True, seed=42, **common
)
test_ds = keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **common)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)


def count_images(folder):
    # Only image files: Windows can leave desktop.ini/Thumbs.db in these folders
    return sum(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(folder))


# Class weights: "disgust" has very few images, so give rare classes more weight
counts = np.array([count_images(os.path.join(train_dir, c)) for c in class_names])
class_weight = {i: counts.sum() / (num_classes * c) for i, c in enumerate(counts)}
print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight.items()})

# ---------- 2. Preprocess + augment ----------
# Augmentation is done in the data pipeline (NOT inside the model)
# so the exported model only contains layers TensorFlow.js understands.
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

train_ds = (
    train_ds.map(lambda x, y: (augment(x / 255.0, training=True), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds =test_ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ---------- 3. Model ----------
def conv_block(x, filters, dropout):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = conv_block(inputs, 32, 0.2)
x = conv_block(x, 64, 0.25)
x = conv_block(x, 128, 0.3)
x = conv_block(x, 256, 0.35)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------- 4. Train ----------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    keras.callbacks.ModelCheckpoint("emotion_model.h5", monitor="val_accuracy", save_best_only=True),
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ---------- 5. Evaluate (first and only time the test set is used) ----------
loss, acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {acc:.3f}")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
except ImportError:
    print("Install scikit-learn for a per-class report.")

# ---------- 6. Save ----------
model.save("emotion_model.h5")
with open("labels.json", "w") as f:
    json.dump(class_names, f)

import tensorflowjs as tfjs  # imported late so training works even if export fails

os.makedirs(args.out, exist_ok=True)
tfjs.converters.save_keras_model(model, args.out)
with open(os.path.join(args.out, "labels.json"), "w") as f:
    json.dump(class_names, f)
# Tells the website a trained model is here (it only loads model.json when this says so)
with open(os.path.join(args.out, "manifest.json"), "w") as f:
    json.dump({"customModel": True, "testAccuracy": round(float(acc), 4)}, f, indent=2)
print(f"\nExported TensorFlow.js model to {os.path.abspath(args.out)}")
