# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 2. Load Fashion-MNIST Data
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

# 3. Prepare Data
X = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
Y = to_categorical(train['label'], num_classes=10)

X_test = test.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
Y_test = test['label'].values
Y_test_cat = to_categorical(Y_test, num_classes=10)

# 4. Train-Validation Split
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)

# 5. Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
valid_datagen = ImageDataGenerator()

# 6. Callbacks
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1)

# 7. CNN Model
model = Sequential([
    Conv2D(64, (3,3), padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(2,2),
    Dropout(0.5),

    Flatten(),
    Dense(256),
    LeakyReLU(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 8. Train Model
history = model.fit(
    train_datagen.flow(X_train, Y_train, batch_size=64),
    epochs=30,
    validation_data=valid_datagen.flow(X_valid, Y_valid),
    callbacks=[early_stop, lr_reduce],
    verbose=2
)

# 9. Evaluate on Test Set
Y_pred_probs = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred_probs, axis=1)

# 10. Classification Report & Confusion Matrix
print("\nAccuracy on Test Set:", accuracy_score(Y_test, Y_pred_classes))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred_classes))

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = confusion_matrix(Y_test, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Fashion-MNIST")
plt.show()

# 11. Accuracy & Loss Curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 12. Show Sample Correct Predictions
correct = [i for i in range(len(Y_test)) if Y_test[i] == Y_pred_classes[i]][:4]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    ax[i//2, i%2].imshow(X_test[correct[i]].reshape(28,28), cmap='gray')
    ax[i//2, i%2].set_title(f"Correct - Predicted: {class_labels[Y_pred_classes[correct[i]]]}")
    ax[i//2, i%2].axis('off')
plt.tight_layout()
plt.show()

# 13. Show Sample Incorrect Predictions
incorrect = [i for i in range(len(Y_test)) if Y_test[i] != Y_pred_classes[i]][:4]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    ax[i//2, i%2].imshow(X_test[incorrect[i]].reshape(28,28), cmap='gray')
    ax[i//2, i%2].set_title(f"Wrong - Pred: {class_labels[Y_pred_classes[incorrect[i]]]}, True: {class_labels[Y_test[incorrect[i]] ]}")
    ax[i//2, i%2].axis('off')
plt.tight_layout()
plt.show()
