import os
import datetime
import pyaudio
import time
import wave

labels = ['Neutral','Disgust','Fear','Sadness', 'Anger', 'Happiness', 'Surprise']
audio_path=os.path.join('FER','audio')
if not os.path.exists(audio_path):
        os.mkdir(audio_path)
for label in labels:
    l_path = os.path.join(audio_path, label)
    if not os.path.exists(l_path):
        os.mkdir(l_path)

#Giving Parameters
# labels=['Fear']
duration = 5
sample_rate=44100
channels=2 
chunk_size=1024 
audio_format=pyaudio.paInt16

name= input('What is your name?\n')
attempt= input('Attmept No.?\n')

for label in labels:
    print("audio for label {}".format(label))
    time.sleep(10)
    print("start now....")

    filename = os.path.join(audio_path,label, f"{label}-{name}-{attempt}.wav")

    print(f"Recording audio for {duration} seconds. Saving to {filename}")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    frames = []

    # Record audio
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    # Stop the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Recording saved to {filename}")