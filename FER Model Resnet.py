import os 
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
import h5py
import random

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

labels = ['Neutral','Disgust','Fear','Sadness', 'Anger', 'Happiness', 'Surprise']
labels_dct={'Neutral': 0,
             'Disgust': 1,
             'Fear': 2,
             'Sadness': 3,
             'Anger': 4,
             'Happiness': 5,
             'Surprise': 6}
img_size=224
root_path=os.path.join('FER','workspace')

train_path=os.path.join(root_path,'train1')
train_aug_path=os.path.join(root_path,'train-aug')
test_path=os.path.join(root_path,'try')
validate_path=os.path.join(root_path,'validate')
model_path=train=os.path.join(root_path,'models')


print(".......................Working on array..........................")
nm_dct={'Neutral': 0,
             'Disgust': 0,
             'Fear': 0,
             'Sadness': 0,
             'Anger': 0,
             'Happiness': 0,
             'Surprise': 0}

lst=os.listdir(train_path)
random.shuffle(lst)
training_arr=[]
cnt=0
for img_nm in lst[:4000]:
    label_nm = img_nm.split('-')[0]
    if label_nm in labels_dct:
        nm_dct[label_nm]+=1
        class_num = labels_dct[label_nm]
        img_array=cv2.imread(os.path.join(train_path,img_nm))
        if img_array is None:
            cnt+=1
            continue
        new_array=cv2.resize(img_array,(img_size,img_size))
        training_arr.append([new_array,class_num])
    else:
        print("error in",img_nm)
        
X,Y=[],[]

for feature,label in training_arr:
    X.append(feature)
    Y.append(label)

X=np.array(X).reshape(-1,img_size,img_size,3)
Y=np.array(Y)
X=X/255.0
print(nm_dct)
print(cnt)

test_arr=[]
for img_nm in os.listdir(test_path):
    label_nm = img_nm.split('-')[0]
    if label_nm in labels_dct:
        class_num = labels_dct[label_nm]
        img_array=cv2.imread(os.path.join(test_path,img_nm))
        new_array=cv2.resize(img_array,(img_size,img_size))
        test_arr.append([new_array,class_num])
    else:
        print("error in",img_nm)    
print(len(test_arr))
random.shuffle(test_arr)
X_test,Y_test=[],[]

for feature,label in test_arr:
    X_test.append(feature)
    Y_test.append(label)

X_test=np.array(X_test).reshape(-1,img_size,img_size,3)
Y_test=np.array(Y_test)
X_test=X_test/255.0
print(X.shape,X_test.shape)
print(Y.shape,Y_test.shape)

print(".......................model dwnld..........................")

model = tf.keras.applications.resnet_v2.ResNet50V2()
base_input=model.layers[0].input
base_output=model.layers[-1].output
final_output=layers.Dense(1024)(base_output)
final_output=layers.Activation('relu')(final_output)
final_output=layers.Dense(512)(final_output)
final_output=layers.Activation('relu')(final_output)
final_output=layers.Dense(256)(final_output)
final_output=layers.Activation('relu')(final_output)
final_output=layers.Dense(128)(final_output)
final_output=layers.Dropout(0.3)(final_output)
final_output=layers.Dense(7,'softmax')(final_output)


new_model =keras.Model(inputs=base_input, outputs=final_output) 
# new_model.summary()
print(".......................Compile..........................")
new_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.00001),metrics=['accuracy'])
history = new_model.fit(X,Y,epochs=40,batch_size=5,validation_data=(X_test,Y_test))
# gpus = tf.config.experimental.list_physical_devices('GPU')
new_model.save('4-resnet50.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()