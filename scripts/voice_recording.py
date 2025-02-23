#%% md
# **Importing modules:**
#%%
from typing import List
import numpy as np
from numpy import ndarray, int16
import sounddevice as sd
from scipy.io.wavfile import write
#%% md
# **Variables:**
#%%
duration: int = 1; # In seconds. Duration of the recording.
fs: int = 22050; # Frequency of recording, 22050 samples per seconds. Continues -> Discrete.
frames: int = duration * fs; # Frame count.


categories: List[str] = ["red", "blue", "off"] 
sample_size: int = 50; # Count for each category.

upper_bound: int = np.iinfo(int16).max # 32767
#%% md
# Recording:
#%%
for category in categories:
    print("Processing category:", category)
    for i in range(sample_size):
        file_name: str = f"{category}_{i}.wav"
        recording: ndarray = sd.rec(frames=frames, samplerate=fs, channels=1)
        sd.wait(ignore_errors=False)
        audio_in_integer_format = (recording * upper_bound).astype(int16)
        write(f"./data/{file_name}", fs, audio_in_integer_format)
        print(f"Recorded for category {category}, index: {i}")
        
print("Finished!")
#%%
