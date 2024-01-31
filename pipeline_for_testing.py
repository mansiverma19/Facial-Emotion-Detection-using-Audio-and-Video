import os
import cv2
import librosa
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

labels = ['Neutral','Disgust','Fear','Sadness', 'Anger', 'Happiness', 'Surprise']

root_path=os.path.join('FER','workspace')
video_path=os.path.join(root_path,'video')
audio_path=os.path.join('FER','audio')
model_path=train=os.path.join(root_path,'models')

# Load pre-trained models (replace these with your actual models)
audio_model = load_model(os.path.join(model_path,'audio-4-89','cnn_model_audio-4(acc=89.5,val=52).h5'))
image_model = load_model(os.path.join(model_path,'3-resnet50-99.3%','3-resnet50.h5'))
faceCascade = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_rectangle(frame,emotion,prediction,bb_lst):
    x=bb_lst[0]
    y=bb_lst[1]
    w=bb_lst[2]
    h=bb_lst[3]
    font_scale=1.5
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    prediction = str(prediction)
    if (emotion=="Neutral"): 
        status = "Neutral"+prediction
        x1,y1, w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1+ h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y1 +int (h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
    elif (emotion=="Disgust"):
        status = "Disgust"+prediction
        x1,y1,w1,h1 =0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y + int (h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    
    elif (emotion=="Fear"): 
        status = "Fear"+prediction
        x1,y1,w1,h1=0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y1+ int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    elif (emotion=="Sadness"):
        status = "Sadness"+prediction
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y1+ int (h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        
    elif (emotion=="Anger"):
        status="Anger"+prediction
        x1,y1, w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10), y1 + int (h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))


    elif (emotion=="Happiness"):
        status = "Happiness"+prediction
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))

    else:
        status = "Surprise"+prediction
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
#         cv2.putText(frame, status, (x1+ int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 0, 255))


def extract_feature(X,sample_rate, mfcc, chroma, mel): 
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0) 
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

# Function to process video data
def process_video(frame):
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMultiScale (gray, scaleFactor=1.1, minNeighbors=4)
    dct = { 0  :  'Neutral',
            1  : 'Disgust',
            2  : 'Fear',
            3  : 'Sadness',
            4  : 'Anger',
            5  : 'Happiness',
            6  : 'Surprise'}
    
    # Adjust this value based on your needs
    min_confidence = 30  

    # Filter faces based on confidence
    filtered_faces = [face for face in faces if face[2] >= min_confidence]

 
    for x,y,w,h in faces:
        roi_gray =gray[y:y+h, x:x+w]
        roi_color = frame [y: y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale (roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi= roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
                
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    Predictions = image_model.predict(final_image)
    
    bb_lst=[x,y,w,h]
    idx = np.argmax(Predictions)
    emotion = dct[idx]
    max_pred = np.max(Predictions)
    #print('video max :',max_pred)
    return emotion,max_pred,bb_lst

# Function to process audio data
def process_audio(audio_data,sr):
#     sr, audio_data = wavfile.read(data)
    result = extract_feature(audio_data,sr, mfcc=True, chroma=True, mel=True)
    result=result.tolist()
    x=[]
    x.append(result)
    x=np.array(x)
    
    dct = { 0  :  'Anger',
            1  : 'Disgust',
            2  : 'Fear',
            3  : 'Happiness',
            4  : 'Neutral',
            5  : 'Sadness',
            6  : 'Surprise'}
    
    # Make predictions using the audio model
    audio_prediction = audio_model.predict(x)
#     print(audio_prediction[0],type(audio_prediction[0]))
    idx=np.argmax(audio_prediction)
    max_pred=np.max(audio_prediction)
    emotion=dct[idx]
    return emotion,max_pred

# Function to integrate video and audio predictions
def pipeline(video_data, audio_data,sr):
    video_emotion,video_prediction,bb_lst= process_video(video_data)
    audio_emotion,audio_prediction = process_audio(audio_data,sr)
    
    print('video_prediction',video_prediction)
    print('audio_prediction',audio_prediction)

    if video_emotion==audio_emotion:
        if video_prediction > audio_prediction:
            return video_emotion,video_prediction,bb_lst
        else:
            return audio_emotion,audio_prediction,bb_lst
    else:
        return video_emotion,video_prediction,bb_lst


import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Set up audio stream parameters
audio_fs = 44100
audio_channels = 1
audio_dtype = pyaudio.paInt16  # Use pyaudio int16 type
audio_chunk_size = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=audio_dtype,
                channels=audio_channels,
                rate=audio_fs,
                input=True,
                frames_per_buffer=audio_chunk_size)

# open video stream
video_capture = cv2.VideoCapture(0)

# Create a real-time audio feature extraction loop
try:
    while True:
       
        ret, frame = video_capture.read()
        
        # Read audio data from the stream
        audio_data = np.frombuffer(stream.read(audio_chunk_size), dtype=np.int16)
        
        # Convert audio data to floating-point format
        audio_data_float = librosa.util.buf_to_float(audio_data, dtype=np.float32)
        #emotion=process_audio(audio_data_float,audio_fs)
        
        emotion,prediction,bb_lst = pipeline(frame,audio_data_float,audio_fs)
       
        print(emotion," :",prediction)
        draw_rectangle(frame,emotion,prediction,bb_lst)
        cv2.imshow('Emotion Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    # Stop the audio stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()


