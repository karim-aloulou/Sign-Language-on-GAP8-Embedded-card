import numpy as np
import pickle
import cv2
import os
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Path to dataset
gestures_path = r'C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\gestures'

# Get image size from the dataset
def get_image_size():
    for root, dirs, files in os.walk(gestures_path):
        for file in files:
            img = cv2.imread(os.path.join(root, file), 0)
            if img is not None:
                print(f"Image size detected: {img.shape}")
                return img.shape
    raise FileNotFoundError("No valid images found in the dataset.")

# Get the number of gesture classes
def get_num_of_classes():
    num_classes = len([name for name in os.listdir(gestures_path) if os.path.isdir(os.path.join(gestures_path, name))])
    print(f"Number of gesture classes detected: {num_classes}")
    return num_classes

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(image_x, image_y, 1), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # Callbacks
    filepath = "cnn_model_keras2.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    callbacks_list = [checkpoint, lr_scheduler, early_stopping]

    return model, callbacks_list


# Normalize labels to start from 0
def normalize_labels(labels, class_mapping):
    """
    Normalize labels based on the provided class mapping.
    """
    # Ensure labels are integers to match the class_mapping keys
    normalized_labels = np.array([class_mapping[label] for label in labels])
    print(f"Normalized labels: {normalized_labels}")
    return normalized_labels


# Training the model
def train():
    # Load data from pickled files
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    print(f"Original train_labels: {np.unique(train_labels)}")
    print(f"Original val_labels: {np.unique(val_labels)}")

    # Dynamically detect classes
    class_mapping = {int(name): idx for idx, name in enumerate(sorted(os.listdir(gestures_path)))}
    print(f"Class mapping: {class_mapping}")

    # Normalize labels using the mapping
    train_labels = normalize_labels(train_labels, class_mapping)
    val_labels = normalize_labels(val_labels, class_mapping)

    print(f"Normalized train_labels: {np.unique(train_labels)}")
    print(f"Normalized val_labels: {np.unique(val_labels)}")

    # Reshape images
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    num_classes = len(class_mapping)

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    print(f"Validation labels shape: {val_labels.shape}")

    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(train_images)


    # Train the model
    model, callbacks_list = cnn_model()
    model.summary()
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=64),
        validation_data=(val_images, val_labels),
        epochs=80,
        callbacks=callbacks_list
    )

    # Evaluate the model
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Analyze performance with confusion matrix
    predictions = np.argmax(model.predict(val_images), axis=1)
    true_labels = np.argmax(val_labels, axis=1)

    print(f"Predictions: {predictions}")
    print(f"True Labels: {true_labels}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    # Load data from pickled files
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    print(f"Original train_labels: {np.unique(train_labels)}")
    print(f"Original val_labels: {np.unique(val_labels)}")

    # Dynamically detect classes
    class_mapping = {int(name): idx for idx, name in enumerate(sorted(os.listdir(gestures_path)))}
    print(f"Class mapping: {class_mapping}")

    # Normalize labels using the mapping
    train_labels = normalize_labels(train_labels, class_mapping)
    val_labels = normalize_labels(val_labels, class_mapping)

    print(f"Normalized train_labels: {np.unique(train_labels)}")
    print(f"Normalized val_labels: {np.unique(val_labels)}")

    # Reshape images
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    num_classes = len(class_mapping)

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    print(f"Validation labels shape: {val_labels.shape}")

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(train_images)

    # Train the model
    model, callbacks_list = cnn_model()
    model.summary()
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=30),
        validation_data=(val_images, val_labels),
        epochs=60,
        callbacks=callbacks_list
    )

    # Evaluate the model
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Analyze performance with confusion matrix
    predictions = np.argmax(model.predict(val_images), axis=1)
    true_labels = np.argmax(val_labels, axis=1)

    print(f"Predictions: {predictions}")
    print(f"True Labels: {true_labels}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

# Get image dimensions
image_x, image_y = get_image_size()

# Train the model
train()

# Clear session to free resources
K.clear_session()
