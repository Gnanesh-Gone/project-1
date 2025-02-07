import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

# Set paths to your dataset directories
data_dir = './sects'
class_names = ['covid', 'normal', 'pneumonia']

# Image size and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Function to split data into train, validation, and test sets
def split_data(data_dir, class_names, test_size=0.2, val_size=0.1):
    filepaths = []
    labels = []

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        cls_files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        filepaths.extend(cls_files)
        labels.extend([cls] * len(cls_files))

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    # Split into train and temp (validation + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filepaths, labels, test_size=test_size + val_size, stratify=labels, random_state=42
    )

    # Split temp into validation and test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=test_size / (test_size + val_size), stratify=temp_labels, random_state=42
    )

    return train_files, train_labels, val_files, val_labels, test_files, test_labels

# Split data
train_files, train_labels, val_files, val_labels, test_files, test_labels = split_data(data_dir, class_names)

# Data generator helper function
def create_generator(files, labels, augment=False):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15 if augment else 0,
        width_shift_range=0.1 if augment else 0,
        height_shift_range=0.1 if augment else 0,
        shear_range=0.1 if augment else 0,
        zoom_range=0.1 if augment else 0,
        horizontal_flip=augment,
        fill_mode='nearest'
    )

    return datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': files, 'class': labels}),
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

# Create generators
train_generator = create_generator(train_files, train_labels, augment=True)
val_generator = create_generator(val_files, val_labels)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_files, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build CNN Model
model = Sequential([
    # Convolutional and Pooling Layers
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten the output to feed into Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to reduce overfitting
    Dense(len(class_names), activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Updated for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1,
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')

# Save the trained model to a .h5 file
model.save('covid_pneumonia_normal_cnn_model.h5')

print("Model saved as covid_pneumonia_normal_cnn_model.h5")
