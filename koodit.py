## Koodit tänne
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
data_dir = "data"


def load_images(data_dir):
    images = []
    labels = []

    for label, folder in enumerate(["no_ship", "ship"]):
        folder_path = os.path.join(data_dir, folder)
        for img_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_path)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((80, 80))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)


# load images
images, labels = load_images(data_dir)

# normalize the pixel values to [0,1]
images = images / 255.0

# convert labels to categorical
labels = to_categorical(labels, num_classes=2)

# First split: Train (80%) and Temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Second split: Validation (10%) and Test (10%) from Temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# spilt for KNN
y_train_knn = y_train.argmax(axis=1)
y_val_knn = y_val.argmax(axis=1)
y_test_knn = y_test.argmax(axis=1)



# Now we have have: tarkista vielä KNN split
# - X_train, y_train for training the CNN
# - X_val, y_val for validating the CNN
# - X_test, y_test for testing the CNN
# - y_train_knn, y_val_knn, y_test_knn for KNN



# data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Build the CNN model
model = Sequential()

# First Conv Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(MaxPooling2D((2, 2)))

# Second Conv Block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third Conv Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(2, activation='softmax'))  # 2 classes

# Display the model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Train the model (without stopper)
#history = model.fit(
#    datagen.flow(X_train, y_train, batch_size=32),
#    epochs=20,
#    validation_data=(X_val, y_val)
#)

#stopper for optimal epoch
early_stopping = EarlyStopping(
    monitor='val_loss',  # or 'val_accuracy'
    patience=3,          # Stop training after 3 epochs without improvement
    restore_best_weights=True  # Restore the weights from the best-performing epoch
)
# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,  # You can set a large number of epochs, say 50 or 100
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['No Ship', 'Ship']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Ship', 'Ship'], yticklabels=['No Ship', 'Ship'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([0, max(history.history['loss']) + 0.5])
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



# ------------------- K-Nearest Neighbors (KNN) Model -------------------

# Flatten images for KNN
X_train_knn = X_train.reshape(X_train.shape[0], -1)
X_val_knn = X_val.reshape(X_val.shape[0], -1)
X_test_knn = X_test.reshape(X_test.shape[0], -1)

# Feature Scaling
scaler = StandardScaler()
X_train_knn = scaler.fit_transform(X_train_knn)
X_val_knn = scaler.transform(X_val_knn)
X_test_knn = scaler.transform(X_test_knn)

# Dimensionality Reduction using PCA
pca = PCA(n_components=50, random_state=42)  # Adjust n_components as needed
X_train_pca = pca.fit_transform(X_train_knn)
X_val_pca = pca.transform(X_val_knn)
X_test_pca = pca.transform(X_test_knn)


print(f"Explained variance by 50 components: {np.sum(pca.explained_variance_ratio_):.2f}")

# Combine training and validation sets for KNN training
X_knn_train = np.vstack((X_train_pca, X_val_pca))
y_knn_train = np.concatenate((y_train_knn, y_val_knn))

# Hyperparameter Tuning for KNN using GridSearchCV
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_knn_train, y_knn_train)

print(f"Best KNN parameters: {grid_search.best_params_}")
print(f"Best KNN cross-validation accuracy: {grid_search.best_score_:.4f}")

# Best KNN model
best_knn = grid_search.best_estimator_


# Evaluate KNN on Test Set
y_pred_knn = best_knn.predict(X_test_pca)
knn_accuracy = accuracy_score(y_test_knn, y_pred_knn)
knn_precision = precision_score(y_test_knn, y_pred_knn)
knn_recall = recall_score(y_test_knn, y_pred_knn)
knn_f1 = f1_score(y_test_knn, y_pred_knn)

print("\nKNN Classification Report:")
print(classification_report(y_test_knn, y_pred_knn, target_names=['No Ship', 'Ship']))

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test_knn, y_pred_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', xticklabels=['No Ship', 'Ship'], yticklabels=['No Ship', 'Ship'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('KNN Confusion Matrix')
plt.show()

# Plot KNN Evaluation Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
cnn_metrics = [
    accuracy_score(y_true, y_pred_classes),
    precision_score(y_true, y_pred_classes),
    recall_score(y_true, y_pred_classes),
    f1_score(y_true, y_pred_classes)
]
knn_metrics = [knn_accuracy, knn_precision, knn_recall, knn_f1]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, cnn_metrics, width, label='CNN', color='blue')
plt.bar(x + width/2, knn_metrics, width, label='KNN', color='green')

plt.ylabel('Scores')
plt.title('Comparison of CNN and KNN Metrics')
plt.xticks(x, metrics)
plt.ylim([0, 1])
plt.legend()
plt.show()
