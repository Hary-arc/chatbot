import sounddevice as sd
import numpy as np
import json
from vosk import Model, KaldiRecognizer

# Load the Vosk model (make sure to download it)
model = Model("model")  # Specify the path to your Vosk model
recognizer = KaldiRecognizer(model, 16000)

# Define the callback to capture audio from the microphone
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    if recognizer.AcceptWaveform(indata):
        result = recognizer.Result()
        print("You said: " + json.loads(result)["text"])
    else:
        partial_result = recognizer.PartialResult()
        print("Partial result: " + json.loads(partial_result)["partial"])

# Setup the sounddevice input stream
with sd.InputStream(callback=callback, channels=1, samplerate=16000):
    print("Say something...")
    sd.sleep(10000) 