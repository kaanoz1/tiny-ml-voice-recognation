#%% md
# **Importing Modules:**
#%%
import pathlib
from pathlib import Path
from typing import final

from scipy.io.wavfile import read
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


from tensorflow.keras.layers import Rescaling, Normalization, TextVectorization
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Normalization, Resizing



import os
from typing import List

#%% md
# Variables:
#%%
fs: int = 22050;
categories: List[str] = ["red", "blue", "off"]
#%% md
# - **Visualizing Raw Wave files Data:**
#%%

off_wav_file = read('./data/off_0.wav')
red_wav_file = read('./data/red_0.wav')
blue_wav_file = read('./data/blue_0.wav')

off_wav_file = off_wav_file[1]
red_wav_file = red_wav_file[1]
blue_wav_file = blue_wav_file[1]

wavefile_plotter = plt.figure(figsize=(18, 10))
off_plot = wavefile_plotter.add_subplot(311)
red_plot = wavefile_plotter.add_subplot(312)
blue_plot = wavefile_plotter.add_subplot(313)

off_plot.plot(off_wav_file)
off_plot.set_ylabel('Amplitude')
off_plot.set_title('Audio: Off')

red_plot.plot(red_wav_file)
red_plot.set_ylabel('Amplitude')
red_plot.set_title('Audio: Red')

blue_plot.plot(blue_wav_file)
blue_plot.set_ylabel('Amplitude')
blue_plot.set_title('Audio: Blue')
#%% md
# **Getting paths of wave files via their labels:**
#%%
data_dir: Path = pathlib.Path('data');

labels = np.array(tf.io.gfile.listdir(str(data_dir)))

audio_files_via_path: list[str] = tf.io.gfile.glob('data/*')

audio_path_files = tf.random.shuffle(audio_files_via_path)
#%% md
# Preprocessing data
#%%
import tensorflow as tf
import os

def extract_label(file_path: tf.Tensor) -> tf.Tensor:
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    label_of_file = tf.strings.split(file_name, "_")[0]
    return label_of_file

def path_to_tensor_beside_label(file_path: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    label_of_file = extract_label(file_path)
    file_tensor = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(file_tensor)
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    return audio_tensor, label_of_file

data_tf_pipeline = tf.data.Dataset.from_tensor_slices(audio_path_files)
labeled_waveform_dataset = data_tf_pipeline.map(path_to_tensor_beside_label)




#%% md
# Visualizing Dataset:
#%%
rows: int = 3
cols: int = 3
n: int = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
for i, (audio, label) in enumerate(labeled_waveform_dataset.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

plt.show()

#%% md
# Spectogram Conversation
#%%
waveform, label = labeled_waveform_dataset.take(1).get_single_element();
spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
spectrogram = tf.abs(spectrogram)
spectrogram = spectrogram[..., tf.newaxis]
label = label.numpy().decode('utf-8')
#%%
from numpy import float64


def plot_spectrogram(spectrogram_param: np.ndarray | tf.Tensor, ax_param: plt.axes):
    eps: float64 = np.finfo(float).eps;

    if len(spectrogram_param.shape) > 2:
        assert len(spectrogram_param.shape) == 3
        spectrogram_param = np.squeeze(spectrogram_param, axis=-1)
    log_spec = np.log(spectrogram_param.T + eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(spectrogram_param), num=width, dtype=int)
    y = range(height)
    ax_param.pcolormesh(x, y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()
#%% md
# **Splitting Data into:**
#  - Training 0.6
#  - Testing 0.2
#   - Validation 0.2
#%%
from typing import Final
from tensorflow.python.types.data import DatasetV2

TRAIN_RATIO: Final[float] = 0.6
TEST_RATIO: Final[float] = 0.2
VALIDATION_RATIO: Final[float] = 0.2

n: int = len(audio_files_via_path)
train_size: int = int(TRAIN_RATIO * n)
test_size: int = int(TEST_RATIO * n)
validation_size: int = int(VALIDATION_RATIO * n)

training_dataset_waveform: DatasetV2 = labeled_waveform_dataset.take(train_size)  # 60% of dataset
test_dataset_waveform: DatasetV2 = labeled_waveform_dataset.skip(train_size).take(test_size)  # 20% of dataset
validation_dataset_waveform: DatasetV2 = labeled_waveform_dataset.skip(train_size + test_size).take(
    validation_size)  # 20% of

#%% md
# Converting Wave paths to labeled Spectograms
#%%
import signal
from tensorflow.python.types.data import DatasetV2


def stft(waveform_par: tf.Tensor) -> tf.Tensor:
    spectrogram_var = tf.signal.stft(waveform_par, frame_length=255, frame_step=124, fft_length=256)
    spectrogram_var = tf.abs(spectrogram_var)
    return spectrogram_var

def waveforms_to_spectrogram(waveform_par: tf.Tensor, label_par: tf.Tensor):
    spectrogram_var = stft(waveform_par)
    spectrogram_var = tf.reshape(spectrogram_var,(129, 176))
    spectrogram_var = tf.expand_dims(spectrogram_var, axis=0)
    label_par = tf.math.argmax(label_par == categories)
    return spectrogram_var, label_par




def convert_waveform_to_spectrogram(dataset_waveform: DatasetV2):
    return dataset_waveform.map(waveforms_to_spectrogram)

training_dataset = convert_waveform_to_spectrogram(training_dataset_waveform)
testing_dataset = convert_waveform_to_spectrogram(test_dataset_waveform)
validation_dataset = convert_waveform_to_spectrogram(validation_dataset_waveform)


shape_of_features = training_dataset.element_spec[0].shape
print(shape_of_features)

#%% md
# **Batching:**
#%%
BATCH_SIZE = 1
train_dataset_batch = training_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)


#%% md
# Building Model:
#%%
norm_layer = Normalization()

norm_layer.adapt(train_dataset_batch.map(lambda x, _: x))

model = models.Sequential()

model.add(layers.Input(shape=shape_of_features))
model.add(Resizing(32, 32))
model.add(norm_layer)
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(categories)))
model.summary()
#%% md
# **Compilation:**
#%%
model.compile(optimizer='adam',
                loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ["accuracy"])

#%%
history = model.fit(train_dataset_batch, validation_data=validation_dataset, epochs=25)
#%% md
# ***Result:***
# accuracy: 0.9990 - loss: 0.0018 - val_accuracy: 0.9333 - val_loss: 0.4264  # For sure, this accuracy rate will undoubtedly go down as adding sample count and variety.
#%% md
# Testing:
# 
#%%
test_spec_arr = [];
test_label_arr = [];

for spectrogram_val, label in testing_dataset:
    test_spec_arr.append(spectrogram_val.numpy())
    test_label_arr.append(label.numpy())

test_spec_arr = np.array(test_spec_arr);
test_label_arr = np.array(test_label_arr);



ith_sample: int = 5;
count_of_samples: int = len(test_spec_arr);

model_prediction = model.predict(test_spec_arr);
actual_label = test_label_arr[ith_sample]


print("Categories:", categories);
print(f"Model Prediction {ith_sample + 1}'th of {count_of_samples}",model_prediction[ith_sample])
print("Actual Label:", categories[actual_label])


#%% md
# Model Saving:
#%%
model.save("models/model_0.keras");