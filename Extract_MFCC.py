import os
import librosa
import numpy as np
import tensorflow as tf

# Model building using tf.keras
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split


# 👇 STEP 1: Feature extraction function (from Step 3)
def extract_features(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


# 👇 STEP 2: Load all files and assign labels (from Step 4)
files_and_labels = [
    ("dataset/help/sample1.wav", 1),
    ("dataset/Non-help/noise.wav", 0),
    ("dataset/Non-help/crowd.wav", 0),
    ("dataset/Non-help/crowd2.wav", 0),
]

X = []
y = []

# 👇 Extract features and store them
for file_path, label in files_and_labels:
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(label)

# 👇 Convert to numpy arrays for TensorFlow
X = np.array(X)
y = np.array(y)

# 👇 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data is ready!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



# 👇 Define the model
model = Sequential([
    Dense(256, input_shape=(40,), activation='relu'),  # 40 = number of MFCC features
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2 output classes: help or not-help
])

# 👇 Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 👇 Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# 👇 Save the trained model to use later
model.save("sos_help_detector.h5")

print("✅ Model training complete and saved!")
