import cv2
import uuid
import os
import time
import tensorflow
import imgaug.augmenters as iaa
from cv2 import VideoWriter_fourcc

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

# Making Directory 

if not os.path.exists(root_path):
    if os.name == 'posix':
        os.mkdir(root_path)
    if os.name == 'nt':
        os.mkdir(root_path)
        
if not os.path.exists(video_path):
    if os.name == 'posix':
        os.mkdir(video_path)
    if os.name == 'nt':
        os.mkdir(video_path)
        
if not os.path.exists(img_path):
    if os.name == 'posix':
        os.mkdir(img_path)
    if os.name == 'nt':
        os.mkdir(img_path)

# Creating different folder for each  emotion in video folder
for label in labels:
    l_path = os.path.join(video_path, label)
    if not os.path.exists(l_path):
        os.mkdir(l_path)
# Creating different folder for each  emotion in image folder
for label in labels:
    l_path = os.path.join(img_path, label)
    if not os.path.exists(l_path):
        os.mkdir(l_path)
# Creating different folder for each  emotion in augmented image folder
for label in labels:
    l_path = os.path.join(aug_img_path, label)
    if not os.path.exists(l_path):
        os.mkdir(l_path)
# Creating different folder for each  emotion in audio folder
for label in labels:
    l_path = os.path.join(audio_path, label)
    if not os.path.exists(l_path):
        os.mkdir(l_path)
# Creating different folder for each  emotion in train folder
if not os.path.exists(train_path):
    if os.name == 'nt':
        os.mkdir(train_path)
if not os.path.exists(train_aug_path):
    if os.name == 'nt':
        os.mkdir(train_aug_path)       
if not os.path.exists(test_path):
    if os.name == 'nt':
        os.mkdir(test_path)

# Define the duration in seconds of the video capture here
capture_duration = 6

print("Age group: 1-10 , 11-20 , 21-30 , 31-40 , 41-50 , 51-60 , 61-70 , 71-80")
age = input('What is your Age group?\n')
name= input('What is your name?\n')

for label in labels:
    print("Video for label {}".format(label))
    time.sleep(15)
    print("start now....")
    webcam = cv2.VideoCapture(0)
    file_path=os.path.join(video_path, label, label+ "-" + age + "-" + name + '.{}.avi'.format(str(uuid.uuid1())))
    video = cv2.VideoWriter(file_path,VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        ret, frame = webcam.read()
        if ret==True:
            cv2.imshow('WEBCAM',frame)
            video.write(frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

# Release everything if job is finished
webcam.release()
video.release()
cv2.destroyAllWindows()

for label in labels:
    # path of each video and image folder
    label_video_path=os.path.join(video_path,label)
    label_img_path=os.path.join(img_path,label)
    
    # getting name of each file in that directory 
    for filename in os.listdir(label_video_path):
        # image name
        img_root_name=str(filename).replace(".avi","")
        # path of the video file
        file_path = os.path.join(label_video_path, filename)
        webcam = cv2.VideoCapture(file_path)
        currentframe = 0
        
        while(True):
            # reading from frame
            ret,frame = webcam.read()
            if ret:
                # if video is still left continue creating images
                name = img_root_name + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                
                # writing the extracted images
                cv2.imwrite(os.path.join(label_img_path,name), frame)
                # show how many frames are created
                currentframe += 1
            else:
                break
webcam.release()
cv2.destroyAllWindows()

aug_lst=[iaa.Fliplr(0.5), # horizontally flip 50% of all images
         iaa.Flipud(0.2), # vertically flip 20% of all images
         iaa.Sharpen(alpha=(0, 0.5),lightness=(0.75, 1.5)),
         iaa.Invert(0.4, per_channel=True),
         iaa.Affine(rotate=(-25, 25))
        ]
for label in labels:
    label_img_path=os.path.join(img_path,label)
    
    # different augmentation
    for category in aug_lst:
        print("category : ",category)
        seq=iaa.Sequential([category])
        
        # iaa.Fliplr(1) -> .Fliplr
        category=str(category)
        sub_nm=category[3:category.find("(")]
        
        i=1
        # all images in that label
        for curr_img_nm in os.listdir(label_img_path):
            # reading image
            if i==1:
                curr=cv2.imread(os.path.join(label_img_path,curr_img_nm))
                new=seq(images=curr)

                # saving new image file in folder
                name=str(curr_img_nm).replace('.jpg',sub_nm+'.jpg')
                cv2.imwrite(os.path.join(aug_img_path,label,name),new)
                i=0
            else:
                i=1
          
# different augmentation
for category in aug_lst:
    print("category : ",category)
    seq=iaa.Sequential([category])

    # iaa.Fliplr(1) -> .Fliplr
    category=str(category)
    sub_nm=category[3:category.find("(")]

    i=1
    # all images in that label
    for curr_img_nm in os.listdir(train_path):
        # reading image
        if i==1:
            curr=cv2.imread(os.path.join(train_path,curr_img_nm))
            new=seq(images=curr)

            # saving new image file in folder
            name=str(curr_img_nm).replace('.jpg',sub_nm+'.jpg')
            cv2.imwrite(os.path.join(train_aug_path,name),new)
            i=0
        else:
            i=1

filename=r'C:/Users/mansi/Videos/sih vid.mp4'
# audio_file_name = str(filename).replace(".avi",".mp3")      
# # path of the video file
# file_path = os.path.join(label_video_path, filename)

# Load the video file
video = VideoFileClip(filename)

# Extract audio from the video
audio = video.audio
if audio==None:
    print("Error")
# Save the extracted audio to a file
audio_file_path = r'C:\Users\mansi\Videos\sih vid.mp3'
audio.write_audiofile(audio_file_name)

# Close the video and audio objects
video.close()
audio.close()

#Cropping data and saving it
for label in labels:
    label_img_path = os.path.join(img_path,label)
    for curr_img_nm in os.listdir(label_img_path):
        frame = cv2.imread(os.path.join(label_img_path,curr_img_nm))
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Create a directory to save the cropped faces
        output_directory = train_path
        os.makedirs(output_directory, exist_ok=True)

        # Crop and save each detected face
        for i, (x, y, w, h) in enumerate(faces):
            roi_color = frame[y:y + h, x:x + w]

            # Save the cropped face
            curr_img_nm=f'{i+1}'+curr_img_nm
            output_path = os.path.join(output_directory,curr_img_nm)
            cv2.imwrite(output_path, roi_color)
            print(label,curr_img_nm)
            # Draw a rectangle around the detected face on the original image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
dct={}
for img_nm in os.listdir(train_path):
    nm=img_nm.split('-')[0]
    if nm in labels:
        if nm not in dct:
            dct[nm]=1
        else:
            dct[nm]+=1
dct