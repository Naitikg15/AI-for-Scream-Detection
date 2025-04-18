import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import smtplib
from email.message import EmailMessage

# Load your trained model
model = load_model("sos_help_detector.h5")

# 🔧 Email config
EMAIL_ADDRESS = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_app_password'  # App password from Gmail
RECEIVER_EMAIL = 'receiver_email@gmail.com'

def send_email_alert():
    msg = EmailMessage()
    msg.set_content("🚨 Emergency! 'HELP' detected 3 times.")
    msg['Subject'] = 'SOS Alert'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
        print("📧 SOS Email sent!")

def record_audio(duration=2, fs=22050):
    print("🎙️ Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc_from_audio(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

help_count = 0

print("🟢 System is running... Say 'HELP' loudly 3 times to trigger SOS.")

while True:
    try:
        audio = record_audio()
        features = extract_mfcc_from_audio(audio)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        if predicted_class == 1:
            help_count += 1
            print(f"🆘 Detected HELP! ({help_count}/3)")
        else:
            print("✅ No help detected.")
            help_count = 0  # reset if interrupted by non-help

        if help_count == 3:
            print("\n🚨 HELP DETECTED 3 TIMES — SENDING SOS...\n")
            send_email_alert()
            break

    except KeyboardInterrupt:
        print("🔴 Program interrupted.")
        break
    except Exception as e:
        print("⚠️ Error:", e)
