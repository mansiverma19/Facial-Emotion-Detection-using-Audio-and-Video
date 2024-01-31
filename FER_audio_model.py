import librosa 
import soundfile
import os, glob, pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

labels = ['Neutral','Disgust','Fear','Sadness', 'Anger', 'Happiness', 'Surprise']

root_path=os.path.join('FER','workspace')
video_path=os.path.join(root_path,'video')
audio_path=os.path.join('FER','audio')
img_path=os.path.join(root_path,'image','collected_img')
aug_img_path=os.path.join(root_path,'image','augmented_img')
train_path=os.path.join(root_path,'train')
train_aug_path=os.path.join(root_path,'train-aug')
test_path=os.path.join(root_path,'test')
validate_path=os.path.join(root_path,'validate')
model_path=train=os.path.join(root_path,'model')

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate=librosa.load(os.path.join(file_name), res_type='kaiser_fast') 
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


def load_data(test_size=0.2):
    x,y=[], []
    for label in os.listdir(audio_path):
        for audio_file in os.listdir(os.path.join(audio_path,label)):
            emotion=audio_file.split('-')[0]
            if emotion not in labels:
                continue
            feature=extract_feature(os.path.join(audio_path,label,audio_file), mfcc=True, chroma=True, mel=True)
#             print(len(feature),feature)
            x.append(feature)
            y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.75, random_state=9)

# Split the Dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.15)

print((x_train.shape, x_test.shape))
print(len(y_train),len(y_test))
x_train[0].shape

# print(lb.fit_transform(y_train))


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

#Encode emotion labels into numbers
# print(lb.fit_transform(y_train))
y_train_lb =np_utils.to_categorical(lb.fit_transform(y_train))
y_test_lb = np_utils.to_categorical(lb.fit_transform(y_test))

# Check out the data
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train_lb.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test_lb.shape}')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled.shape,x_test_scaled.shape)

model=MLPClassifier (alpha=0.001, batch_size=4, epsilon=1e-08, hidden_layer_sizes=(800,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train_lb)

# Predict for the test set
y_train_pred=model.predict(x_train)
accuracy=accuracy_score (y_true=y_train_lb, y_pred=y_train_pred)
print("Train Accuracy: {:.2f}%".format(accuracy*100))

y_test_pred=model.predict(x_test)
accuracy=accuracy_score (y_true=y_test_lb, y_pred=y_test_pred)
print("Test Accuracy: {:.2f}%".format(accuracy*100))

from sklearn.metrics import classification_report 

lst=lb.inverse_transform([0,1,2,3,4,5,6])
for i in range(0,7):
    print(i," : ",lst[i])
print(classification_report (y_test_lb,y_pred))

from joblib import Parallel, delayed 
import joblib 

joblib.dump(model, 'MLP-1.pkl') 
  
# Load the model from the file 
MLP_model = joblib.load('MLP-1.pkl')  
MLP_model.predict(x_test)  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, Activation, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import h5py

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

CNN_model = Sequential()

#Build first layer
CNN_model.add(Conv1D(16, 5,padding='same',input_shape=(180, 1), activation='relu'))
CNN_model.add(BatchNormalization())

#Build second layer
CNN_model.add(Conv1D(32, 5,padding='same',activation='relu'))
CNN_model.add(BatchNormalization())

#Build third layer
CNN_model.add(Conv1D(64, 5,padding='same',activation='relu'))
CNN_model.add(BatchNormalization())

#Add dropout
CNN_model.add(Dropout(0.1))

#Build forth layer
CNN_model.add(Conv1D(128, 5,padding='same',activation='relu'))
CNN_model.add(BatchNormalization())

#Build forth layer
CNN_model.add(Conv1D(256, 5,padding='same',activation='relu'))
CNN_model.add(BatchNormalization())

# CNN_model.add(MaxPooling1D(pool_size=5))

#Flatten 
CNN_model.add(Flatten())

CNN_model.add(Dense(128, activation ='relu'))
CNN_model.add(Dropout(0.4))
CNN_model.add(Dense(64, activation ='relu'))
CNN_model.add(Dense(7, activation='softmax'))

CNN_model.compile(loss = 'categorical_crossentropy',
                  optimizer =Adam(lr=0.000001),
                  metrics = ['accuracy'])

cnn_results = CNN_model.fit(x_train, y_train_lb,
              batch_size = 32,
              epochs = 400,
              verbose = 1,
              validation_data = (x_test, y_test_lb))

from matplotlib import pyplot as plt
# Accuracy
# epochs = range(1, len(acc) + 1)
plt.plot(cnn_results.history['accuracy'])
plt.plot(cnn_results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# LOSS
plt.plot(cnn_results.history['loss'])
plt.plot(cnn_results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

CNN_model.save('cnn_model_audio-4(acc=89.5,val=52).h5')


from sklearn import svm
lb1 = LabelEncoder()

#Encode emotion labels into numbers
# print(lb.fit_transform(y_train))
y_train_lb1 =lb1.fit_transform(y_train)
y_test_lb1 = lb1.fit_transform(y_test)

# Check out the data
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train_lb1.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test_lb1.shape}')


svm_classifier1 = svm.SVC(kernel='linear')  # You can use 'rbf' or other kernels as well

# Train the SVM classifier
# y_train_lb1=y_train_lb.reshape(-1)
# print(y_train_lb1)
svm_classifier1.fit(x_train, y_train_lb1)

# Make predictions on the test set
y_pred1 = svm_classifier1.predict(x_train)

# Evaluate the accuracy
accuracy1 = accuracy_score(y_train_lb1, y_pred1)
print(f"Train Accuracy: {accuracy1 * 100:.2f}%")

# Make predictions on the test set
y_pred2 = svm_classifier1.predict(x_test)

# Evaluate the accuracy
accuracy2 = accuracy_score(y_test_lb1, y_pred2)
print(f"Test Accuracy: {accuracy2 * 100:.2f}%")


svm_classifier2 = svm.SVC(kernel='linear')
svm_classifier2.fit(x_train_scaled, y_train_lb1)

# Make predictions on the test set
y_pred1 = svm_classifier2.predict(x_train_scaled)

# Evaluate the accuracy
accuracy1 = accuracy_score(y_train_lb1, y_pred1)
print(f"Train Accuracy: {accuracy1 * 100:.2f}%")

# Make predictions on the test set
y_pred2 = svm_classifier2.predict(x_test_scaled)

# Evaluate the accuracy
accuracy2 = accuracy_score(y_test_lb1, y_pred2)
print(f"Test Accuracy: {accuracy2 * 100:.2f}%")