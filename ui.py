import joblib
from pydub import AudioSegment
import librosa
import soundfile

import joblib
import numpy as np
import tkinter.messagebox as msg

from PIL import Image, ImageTk
import PIL.Image
from itertools import count, cycle


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result
##msg.showinfo("Prediction",f'The emotion detected by the person is {prediction}')

# Writing different model files to file

filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)  # loading the model file from the storage
##from tensorflow import keras
##loaded_modelmodel = keras.models.load_model(filename)

##gui :(

import tkinter as tk

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)

    return filename

class ImageLabel(tk.Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """

    def load(self, im):
        if isinstance(im, str):
            im = PIL.Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)

from tkinter import *
import pygame
from tkinter import filedialog
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
root = tk.Tk()

root.title('SER')
apps=[]
lbl = ImageLabel(root)
lbl.pack()
lbl.load('ttt.gif')

def upload_file():
    filename = filedialog.askopenfilename()

    return filename

pygame.mixer.init()
def play():
    file1=upload_file()
    pygame.mixer.music.load(file1)
    pygame.mixer.music.play(loops=0)
    return file1

def predict():
    final=play()
    feature = extract_feature(final, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    result = loaded_model.predict(feature)
    a = tk.Label(root, text=result).place(x=1, y=100)

def Record():
    freq = 44100
    duration = 3
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=2)
    sd.wait()
    wv.write("recording1.wav", recording, freq, sampwidth=2)
    stereo_audio = AudioSegment.from_file("recording1.wav")

    mono_audios = stereo_audio.split_to_mono()
    mono_left = mono_audios[0].export(
        "example voices\\mono_left.wav",
        format="wav")


RUN1=Button(root,text="Predict", padx=15,pady=10 , fg="White",bg="BLUE",command =predict).place(x=1, y=1)
RUN2=Button(root,text="Record", padx=15,pady=10 , fg="White",bg="BLUE",command =Record).place(x=30, y=50)
exit_button = Button(root, text="Exit", padx=5,pady=5 , fg="white",bg="red", command=root.destroy).place(x=765, y=567)


root.mainloop()