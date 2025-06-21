# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Input, Model, Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_1
from tensorflow.keras.layers import RandomFlip, RandomRotation, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- CONSTANTS ---
IMAGES_FP = './images'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 16

# --- LOADING DATA ---
image_paths = glob.glob(os.path.join(IMAGES_FP, '*.jpg'))
image_names = [os.path.basename(file) for file in image_paths]

# Label extraction
labels_raw = [''.join(name.split('_')[:-1]) for name in image_names]

# Label Encoding
def label_encode(label):
    label_dict = {
        'Abyssinian': 0,
        'Bengal': 1,
        'Birman': 2,
        'Bombay': 3,
        'BritishShorthair': 4,
        'EgyptianMau': 5,
        'Americanbulldog': 6,
        'Americanpitbullterrier': 7,
        'bassethound': 8,
        'beagle': 9,
        'boxer': 10,
        'chihuahua': 11,
        'englishcockerspaniel': 12,
        'germanshorthaired': 13,
        'greatpyrenees': 14,
        'havanese': 15  # Example (add missing classes if needed)
    }
    return label_dict.get(label, None)

features = []
labels = []

for path, label_text in zip(image_paths, labels_raw):
    label_encoded = label_encode(label_text.replace(' ', '').lower())
    if label_encoded is not None:
        img = load_img(path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img, dtype='uint8')
        img_array = tf.image.resize_with_pad(img_array, *IMAGE_SIZE).numpy().astype('uint8')
        features.append(img_array)
        labels.append(label_encoded)

features_array = np.array(features)
labels_array = np.array(labels)

# One-hot encode labels
labels_one_hot = to_categorical(labels_array, num_classes=NUM_CLASSES)

# --- DATA SPLITTING ---
x_train, x_temp, y_train, y_temp = train_test_split(features_array, labels_one_hot, test_size=0.4, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)

print(f"Train set size: {x_train.shape}, Validation set size: {x_val.shape}, Test set size: {x_test.shape}")

# --- MODEL BUILDING ---
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
])

base_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = pp_1(x)
x = base_model(x, training=False)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- TRAINING ---
EPOCHS = 10

history = model.fit(
    x=x_train, y=y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS
)

# --- PLOTTING TRAINING HISTORY ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# --- EVALUATION ---
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# --- PREDICTIONS ---
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# OPTIONAL: You can also check classification report if needed
from sklearn.metrics import classification_report
print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes))
