import librosa
import soundfile
import os, glob

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import wave as wav

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
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
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'sad', 'angry']


# Load the data and extract features for each sound file
def load_data(test_size):
    x, y = [], []

    for file in glob.glob("dataset/Actor_*/*.wav"):
        file_name = os.path.basename(file)

        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.2)


# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
##hidden layer =No. Of neurons in the hidden layer
##learning rate adpative= adaptive’ keeps the learning rate constant
##max_itration= how many times each data point will be used
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)


# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import accuracy_score, f1_score

print("f1_score :",f1_score(y_test, y_pred, average=None))

import pandas as pd
df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
print(df.head(10))



