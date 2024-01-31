import librosa
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Example usage
audio_path = r'FER\audio\Anger\Anger-Actor_01-4.wav'
labels = ['Neutral','Disgust','Fear','Sadness', 'Anger', 'Happiness', 'Surprise']

def plot_tempo(audio_file_path,label):
    # Load audio file
    y, sr = librosa.load(audio_file_path)
    # Estimate tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # Plot onset envelope and tempo
    plt.figure(figsize=(12, 6))
    # Plot onset envelope
    plt.subplot(2, 1, 1)
    plt.plot(librosa.times_like(onset_env), onset_env, label='Onset Strength')
    plt.title('Onset Envelope')
    plt.legend()
    # Plot tempo
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(librosa.times_like(onset_env), 0, y.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.title(f'{label} -Tempo: {tempo:.2f} BPM')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pitch(audio_path,label, frame_size=2048, hop_size=512, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr) 
    # Calculate the short-time Fourier transform (STFT)
    D = librosa.stft(y, n_fft=frame_size, hop_length=hop_size)
    # Calculate the magnitude spectrum
    magnitude = np.abs(D)
    # Find the index of the maximum magnitude in each frame
    max_magnitude_index = np.argmax(magnitude, axis=0)
    # Convert the index to frequency in Hz
    frequencies = librosa.core.fft_frequencies(sr=sr, n_fft=frame_size)
    pitch_freqs = frequencies[max_magnitude_index]
    # Convert frequency to pitch in Hz
    pitch = librosa.hz_to_midi(pitch_freqs)
    print(pitch.shape)
     # Plot the pitch
    times = librosa.times_like(pitch, sr=sr, hop_length=hop_size)
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.plot(times, pitch, label='Pitch (Hz)', color='r', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Pitch Detection '+label)
    plt.legend()
    plt.show()

import numpy as np

def plot_chroma_pitch(audio_path, sr=22050, n_chroma=12):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    # Plot the chroma feature
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.show()
    
def plot_mfcc(audio_path,label, sr=22050, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Plot the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.title('MFCCs '+label)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()
def plot_amplitude(audio_path,label):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    # Compute the amplitude envelope
    amplitude_envelope = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # Plot the amplitude envelope
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(amplitude_envelope, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Amplitude'+label)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

import os
audio_path=os.path.join('FER','audio')

for label in labels:
    for audio in os.listdir(os.path.join(audio_path,label)):
        path=os.path.join(audio_path,label,audio)
        plot_pitch(path,label)
        break