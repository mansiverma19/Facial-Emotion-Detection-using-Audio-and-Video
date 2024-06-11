# Facial Emotion Recognition

Facial Emotion Recognition is a system designed to identify 7 distinct emotions from both audio and video data. The system leverages the ResNet50 architecture along with data augmentation techniques to ensure high accuracy and robustness.

## Technology Stack

- *Programming Language*: Python
- *Libraries*: OpenCV, TensorFlow
- *Model Architecture*: Convolutional Neural Network (CNN) using ResNet50

## Key Features

- Identification of 7 different emotions from facial expressions.
- Integration of both audio and video data for comprehensive emotion recognition.
- Custom dataset creation and extensive data cleaning.
- Data augmentation to enhance model robustness.
- Achieved 89% accuracy in real-world scenarios.

## Key Responsibilities

- Developed a Facial Emotion Recognition system using ResNet50.
- Built a custom dataset incorporating diverse facial expressions.
- Cleaned data to remove noise and inconsistencies.
- Applied data augmentation techniques for better generalization.
- Trained the model to accurately classify emotions.
- Implemented the system to process audio and video inputs for comprehensive emotion recognition.

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4+
- TensorFlow 2.4+

### Setup

1. *Clone the repository*:
    bash
    git clone https://github.com/yourusername/facial-emotion-recognition.git
    cd facial-emotion-recognition
    

2. *Create a virtual environment and activate it*:
    bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    

3. *Install the dependencies*:
    bash
    pip install -r requirements.txt
    

4. *Download the pretrained ResNet50 model*:
    bash
    # Example code to download the model; replace with actual commands if necessary
    wget https://path/to/pretrained/resnet50/model
    

## Custom Dataset

### Building the Dataset

1. *Collect Images and Audio*: Gather a diverse set of facial images and corresponding audio clips representing 7 emotions: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Neutral.

2. *Data Cleaning*: 
    python
    import pandas as pd
    from sklearn.utils import shuffle

    # Load the dataset
    data = pd.read_csv('path/to/dataset.csv')

    # Remove duplicates and null values
    data = data.drop_duplicates().dropna()

    # Shuffle the dataset
    data = shuffle(data)
    

3. *Data Augmentation*:
    python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Example usage with a sample image
    img = load_img('path/to/image.jpg')  # Load image
    x = img_to_array(img)  # Convert to array
    x = x.reshape((1,) + x.shape)  # Reshape

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # Generate 20 augmented images
    

## Training the Model

1. *Model Definition*:
    python
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    

2. *Training*:
    python
    model.fit(
        train_data,
        epochs=25,
        validation_data=val_data,
        steps_per_epoch=len(train_data) // batch_size,
        validation_steps=len(val_data) // batch_size
    )
    

## Running the System

1. *Real-Time Emotion Recognition*:
    python
    import cv2
    from tensorflow.keras.models import load_model

    emotion_model = load_model('path/to/emotion_model.h5')

    def predict_emotion(frame):
        face = preprocess_face(frame)  # Function to preprocess face
        emotion = emotion_model.predict(face)
        return emotion

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotion = predict_emotion(frame)
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

## Evaluation

Evaluate the model on the test dataset to ensure it generalizes well to unseen data.

python
results = model.evaluate(test_data)
print(f'Test Accuracy: {results[1]*100:.2f}%')


## Contributing

1. Fork the repository.
2. Create your feature branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.
