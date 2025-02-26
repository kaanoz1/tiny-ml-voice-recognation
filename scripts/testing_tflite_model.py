#%%
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy import signal

from numpy import ndarray, int16, float32
from typing import List
#%%
duration: int = 1;  # In seconds. Duration of the recording.
fs: int = 22050;  # Frequency of recording, 22050 samples per seconds. Continues -> Discrete.
frames: int = duration * fs;  # Frame count.

categories: List[str] = ["red", "blue", "off"]
#%%
def get_waveform_file(file_name: str) -> ndarray:
    file_path: str = f"./data/{file_name}"
    file_tensor = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(file_tensor)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    return audio_tensor
#%%
print("Speak Now")
recording: ndarray = sd.rec(frames=frames, samplerate=fs, channels=1)
sd.wait(ignore_errors=False)

audio = np.squeeze(recording).astype(np.float32)
audio = get_waveform_file("off_5.wav")
spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=124, fft_length=256)

spectrogram = tf.abs(spectrogram)
spectrogram = tf.reshape(spectrogram, (1, 1, 129, 176))

tf_model = tf.lite.Interpreter(model_path="models/model_0.tflite")
input_details = tf_model.get_input_details()
output_details = tf_model.get_output_details()

print("Spectrogram Shape: ", spectrogram.shape)
print("Input Shape: ", input_details[0]["shape"])

## Model Predicting:

tf_model.allocate_tensors()
tf_model.set_tensor(input_details[0]["index"], spectrogram)
tf_model.invoke()

tf_model_prediction_coefficients = tf_model.get_tensor(output_details[0]["index"])

tf_model_prediction = categories[tf.argmax(tf_model_prediction_coefficients, axis=1).numpy()[0]]


print("Categories: ", categories)
print("Coefficients: ", tf_model_prediction_coefficients)
print("Model Prediction: ", tf_model_prediction)

print("Finished!")
