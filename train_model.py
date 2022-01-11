import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the variables

gestures = {'A_': 'A', 'pa': 'B', 'C_': 'C', 'D_': 'D', 'fi': 'E', 'ok': "F", 'G_': 'G', 'I_': 'I', 'L_': 'L',
            'pe': 'V'}
gestures_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'I': 7, 'L': 8, 'V': 9}
gesture_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L', 9: 'V'}

image_path = 'data'
models_path = 'models/saved_model.hdf5'
rgb = False
imageSize = 224


# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = np.array(img)
    return img


# Xu ly du lieu dau vao
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype='float32')
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,) * 3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data


# Ham duuyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                if file[0:2] in gestures:
                    gesture_name = gestures[file[0:2]]
                    print(gesture_name)
                    print(gestures_map[gesture_name])
                    y_data.append(gestures_map[gesture_name])
                    X_data.append(process_image(path))

            else:
                continue

    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data


# Load du lieu vao X va Y
X_data, y_data = walk_file_tree(image_path)

# Split the data into training set and test set with proportion 80:20
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

# Setting the checkpoint to get the best model
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',
                               restore_best_weights=True)

# Initial the model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizers1 = optimizers.Adam()
base_model = model1

# Adding more layer
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# These lines of code just training for the extra layer we have just added
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

# Save the model after training into file
model.save('models/mymodel.h5')
