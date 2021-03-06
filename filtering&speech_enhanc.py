# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F25bdwHNJV-Mvvj40d5m0r3p2EreB7fz
"""

pip install --upgrade scipy

pip install --user --upgrade scipy

pip install pydub

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Station/sp05_station_sn5.wav')
noisy_s = noisy_speech.get_array_of_samples() # samples x(t)
noisy_f = noisy_speech.frame_rate

plt.figure(figsize = (15, 5))
plt.plot(noisy_s)
plt.xlabel('Samples')
plt.ylabel('Amplitude')

freq_range = 2048
#window size: the number of samples per frame
#each frame is of 30ms
win_length = int(noisy_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
#windowing technique
window = 'hann'
noisy_S = librosa.stft(np.float32(noisy_s),
n_fft = freq_range,window = window,
hop_length = hop_length,
win_length = win_length)
plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
noisy_Sdb = librosa.amplitude_to_db(abs(noisy_S))
librosa.display.specshow(
#spectrogram
noisy_Sdb,
#sampling rate
sr = noisy_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

from scipy import signal
#order
order = 10
sampling_freq = noisy_f
#cut-off frequency. This can be an array if band-pass filter is used
#this must be within 0 and cutoff_freq/2
cutoff_freq = 1000
#filter type, e.g., 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
filter_type = 'lowpass'
#filter
h = signal.butter(N = order,
fs = sampling_freq,
Wn = cutoff_freq,
btype = filter_type,
analog = False,
output = 'sos')

filtered_s = signal.sosfilt(h, noisy_s)

import array
import pydub
from pydub import AudioSegment
filtered_s_audio = pydub.AudioSegment(
#raw data
data = array.array(noisy_speech.array_type, np.float16(filtered_s)),
#2 bytes = 16 bit samples
sample_width = 2,
#frame rate
frame_rate = noisy_f,
#channels = 1 for mono and 2 for stereo
channels = 1)
filtered_s_audio.export('sp01_station_sn5_lowpass.wav', format = 'wav')

filtered_S = librosa.stft(np.float32(filtered_s),
n_fft = freq_range,window = window,
hop_length = hop_length,
win_length = win_length)
from matplotlib import pyplot as plt
plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
F_hat_db = librosa.amplitude_to_db(abs(filtered_S))
librosa.display.specshow(

F_hat_db,
#sampling rate
sr = noisy_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

from scipy import signal
#order
order = 10
sampling_freq = noisy_f
#cut-off frequency. This can be an array if band-pass filter is used
#this must be within 0 and cutoff_freq/2
cutoff_freq = 200
#filter type, e.g., 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
filter_type = 'highpass'
#filter
h = signal.butter(N = order,
fs = sampling_freq,
Wn = cutoff_freq,
btype = filter_type,
analog = False,
output = 'sos')

filtered_s = signal.sosfilt(h, noisy_s)

import array
import pydub
from pydub import AudioSegment
filtered_s_audio = pydub.AudioSegment(
#raw data
data = array.array(noisy_speech.array_type, np.float16(filtered_s)),
#2 bytes = 16 bit samples
sample_width = 2,
#frame rate
frame_rate = noisy_f,
#channels = 1 for mono and 2 for stereo
channels = 1)
filtered_s_audio.export('sp01_station_sn5_highpass.wav', format = 'wav')

filtered_S = librosa.stft(np.float32(filtered_s),
n_fft = freq_range,window = window,
hop_length = hop_length,
win_length = win_length)
from matplotlib import pyplot as plt
plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
F_hat_db = librosa.amplitude_to_db(abs(filtered_S))
librosa.display.specshow(

F_hat_db,
#sampling rate
sr = noisy_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

from scipy import signal
#order
order = 10
sampling_freq = noisy_f
#cut-off frequency. This can be an array if band-pass filter is used
#this must be within 0 and cutoff_freq/2
cutoff_freq = [200,1000]
#filter type, e.g., 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
filter_type = 'bandpass'
#filter
h = signal.butter(N = order,
fs = sampling_freq,
Wn = cutoff_freq,
btype = filter_type,
analog = False,
output = 'sos')

filtered_s = signal.sosfilt(h, noisy_s)

import array
import pydub
from pydub import AudioSegment
filtered_s_audio = pydub.AudioSegment(
#raw data
data = array.array(noisy_speech.array_type, np.float16(filtered_s)),
#2 bytes = 16 bit samples
sample_width = 2,
#frame rate
frame_rate = noisy_f,
#channels = 1 for mono and 2 for stereo
channels = 1)
filtered_s_audio.export('sp01_station_sn5_bandpass.wav', format = 'wav')

filtered_S = librosa.stft(np.float32(filtered_s),
n_fft = freq_range,window = window,
hop_length = hop_length,
win_length = win_length)
from matplotlib import pyplot as plt
plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
F_hat_db = librosa.amplitude_to_db(abs(filtered_S))
librosa.display.specshow(

F_hat_db,
#sampling rate
sr = noisy_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)







from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Station/sp05_station_sn5.wav')
y = noisy_speech.get_array_of_samples() # samples x(t)
y_f = noisy_speech.frame_rate
win_length = int(y_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
Y = librosa.stft(np.float32(y),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_Y = abs(Y)

from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Station/sp05_station_sn5.wav')
d = noisy_speech.get_array_of_samples() # samples x(t)
d_f = noisy_speech.frame_rate
win_length = int(d_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
D = librosa.stft(np.float32(d),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_D = abs(D)

means_mag_D = np.mean(mag_D, axis = 1)

H = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
W = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(H.shape[0]):
  for t in range(H.shape[1]):
    H[k][t] =np.sqrt(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))
print(H)

for k in range(W.shape[0]):
  for t in range(W.shape[1]):
    W[k][t] =(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))
print(W)

S_hat = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(H.shape[0]):
  for t in range(H.shape[1]):
    S_hat[k][t] = H[k][t] * Y[k][t]
print(S_hat)

S_hat1 = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(W.shape[0]):
  for t in range(W.shape[1]):
    S_hat1[k][t] = W[k][t] * Y[k][t]
print(S_hat1)

win_length = int(y_f * 0.03)
hop_length = int(win_length / 2)
s_hat = librosa.istft(S_hat, win_length = win_length, hop_length = hop_length, length = len(y))
s_hat1 = librosa.istft(S_hat1, win_length = win_length, hop_length = hop_length, length = len(y))
print(s_hat)
print(s_hat1)

import array
import pydub
from pydub import AudioSegment
s_hat_audio = pydub.AudioSegment(data=array.array(noisy_speech.array_type,np.float16(s_hat) ),
                                 sample_width=2,frame_rate = y_f,channels=1)
s_hat_audio.export('sp01_station_sn5_Wiener.wav', format = 'wav')

import array
import pydub
from pydub import AudioSegment
s_hat1_audio = pydub.AudioSegment(data=array.array(noisy_speech.array_type,np.float16(s_hat1) ),
                                 sample_width=2,frame_rate = y_f,channels=1)
s_hat1_audio.export('sp01_station_sn5_Wiener_1.wav', format = 'wav')

from matplotlib import pyplot as plt
plt.figure(figsize = (20, 10))
#convert the amplitude to decibels, just for illustration purpose
S_hat_db = librosa.amplitude_to_db(abs(S_hat))
librosa.display.specshow(

S_hat_db,
#sampling rate
sr = y_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

plt.figure(figsize = (20  , 10))
#convert the amplitude to decibels, just for illustration purpose
S_hat_db_1= librosa.amplitude_to_db(abs(S_hat1))
librosa.display.specshow(

S_hat_db_1,
#sampling rate
sr = y_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

#Visualising the clean signal

c_speech = AudioSegment.from_wav('/content/sample_data/CleanSignal/sp01.wav')
c_s = c_speech.get_array_of_samples() # samples x(t)
c_f = c_speech.frame_rate

freq_range = 2048
#window size: the number of samples per frame
#each frame is of 30ms
win_length = int(c_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
#windowing technique
window = 'hann'
c_S = librosa.stft(np.float32(c_s),
n_fft = freq_range,window = window,
hop_length = hop_length,
win_length = win_length)
plt.figure(figsize = (15, 5))
#convert the amplitude to decibels, just for illustration purpose
c_Sdb = librosa.amplitude_to_db(abs(c_S))
librosa.display.specshow(
#spectrogram
c_Sdb,
#sampling rate
sr = c_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

#Spectral subtraction on supplied Noisy Signal

from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Babble/sp01_babble_sn5.wav')
y = noisy_speech.get_array_of_samples() # samples x(t)
y_f = noisy_speech.frame_rate
win_length = int(y_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
Y = librosa.stft(np.float32(y),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_Y = abs(Y)

from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Babble/sp01_babble_sn5.wav')
d = noisy_speech.get_array_of_samples() # samples x(t)
d_f = noisy_speech.frame_rate
win_length = int(d_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
D = librosa.stft(np.float32(d),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_D = abs(D)

means_mag_D = np.mean(mag_D, axis = 1)

H = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)

for k in range(H.shape[0]):
  for t in range(H.shape[1]):
    H[k][t] =np.sqrt(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))
print(H)

S_hat = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(H.shape[0]):
  for t in range(H.shape[1]):
    S_hat[k][t] = H[k][t] * Y[k][t]
print(S_hat)

win_length = int(y_f * 0.03)
hop_length = int(win_length / 2)
s_hat = librosa.istft(S_hat, win_length = win_length, hop_length = hop_length, length = len(y))

import array
import pydub
from pydub import AudioSegment
s_hat_audio = pydub.AudioSegment(data=array.array(noisy_speech.array_type,np.float16(s_hat) ),
                                 sample_width=2,frame_rate = y_f,channels=1)
s_hat_audio.export('sp01_babble_sn5_Wiener.wav', format = 'wav')

from matplotlib import pyplot as plt
plt.figure(figsize = (20, 10))
#convert the amplitude to decibels, just for illustration purpose
S_hat_db = librosa.amplitude_to_db(abs(S_hat))
librosa.display.specshow(

S_hat_db,
#sampling rate
sr = y_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Babble/sp01_babble_sn5.wav')
y = noisy_speech.get_array_of_samples() # samples x(t)
y_f = noisy_speech.frame_rate
win_length = int(y_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
Y = librosa.stft(np.float32(y),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_Y = abs(Y)

from pydub import AudioSegment
noisy_speech = AudioSegment.from_wav('/content/sample_data/NoisySignal/Babble/sp01_babble_sn5.wav')
d = noisy_speech.get_array_of_samples() # samples x(t)
d_f = noisy_speech.frame_rate
win_length = int(d_f * 0.03)
#number of samples between two consecutive frames
#by default, hop_length = win_length / 4
hop_length = int(win_length / 2)
D = librosa.stft(np.float32(d),
n_fft = 2048,
window = 'hann',
hop_length = hop_length,
win_length = win_length)
mag_D = abs(D)

means_mag_D = np.mean(mag_D, axis = 1)

W = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(W.shape[0]):
  for t in range(W.shape[1]):
    W[k][t] =(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))
print(W)

S_hat1 = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
for k in range(W.shape[0]):
  for t in range(W.shape[1]):
    S_hat1[k][t] = W[k][t] * Y[k][t]
print(S_hat1)

win_length = int(y_f * 0.03)
hop_length = int(win_length / 2)

s_hat1 = librosa.istft(S_hat1, win_length = win_length, hop_length = hop_length, length = len(y))

print(s_hat1)

import array
import pydub
from pydub import AudioSegment
s_hat1_audio = pydub.AudioSegment(data=array.array(noisy_speech.array_type,np.float16(s_hat1) ),
                                 sample_width=2,frame_rate = y_f,channels=1)
s_hat1_audio.export('sp01_babble_sn5_Wiener_1.wav', format = 'wav')

plt.figure(figsize = (20  , 10))
#convert the amplitude to decibels, just for illustration purpose
S_hat_db_1= librosa.amplitude_to_db(abs(S_hat1))
librosa.display.specshow(

S_hat_db_1,
#sampling rate
sr = y_f,
#label for horizontal axis
x_axis = 'time',
#presentation scale
y_axis = 'linear',
#hop_lenght
hop_length = hop_length)

