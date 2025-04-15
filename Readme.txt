This folder contains the AI Model for Scream Detection by screaming "Help" which is designed for SafeMitra Project in Hacksagon 2k25.
This model uses Tensorflow and Python to work.
Currently, It is only limited to detecting and recognizing the "Help" Screams from audio files.k

You can test this model by:-
1. Record your voice by screaming "Help" using microphone.
2. Remember, the file type should be .wav only.
3. COpy the path of your audio file and paste it in the required place in the predict_sos.py
4. Run the file.
5. The terminal will respond with something like "Detected Help, Sending SOS."

Note: You may need to install some pip libraries like:- 

pip install librosa

pip install soundfile audioread

pip install numpy

pip install matplotlib

pip install tensorflow
