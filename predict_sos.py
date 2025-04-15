import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# 👇 Load the trained model
model = load_model("sos_help_detector.h5")

# 👇 Same feature extraction as before
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print("Error:", e)
        return None

# 👇 List of 3 test clips (simulating real-time detection)
test_files = [
    "test_audio/help_test1.wav",#Paste the path to your test audios here
    "test_audio/help_test2.wav",
    "test_audio/help_test3.wav",#Paste the path to your test audios here
]

help_count = 0

for file in test_files:
    print(f"\n🎧 Analyzing: {file}")
    features = extract_features(file)
    if features is None:
        continue

    features = np.expand_dims(features, axis=0)  # Reshape for model
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    if predicted_class == 1:
        print("🆘 Detected: HELP!")
        help_count += 1
    else:
        print("✅ Sound is normal.")
        help_count = 0  # Reset count if one fails

    # 👇 If 3 consecutive "help" detections → trigger SOS
    if help_count == 3:
        print("\n🚨 SOS Triggered! Sending emergency alert!\n")
        # Here you can add actual alert logic like sending an SMS, email, or notification
        break
