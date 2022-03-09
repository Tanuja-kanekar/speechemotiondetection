import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('final1.pkl','rb'))

def main():
    st.title("Hello!! I am Speech Emotion Recognizer")
    st.markdown("#### Send the file and I will try my best to predict the emotion")
    audio_file = st.file_uploader("Upload audio file", type=['wav'])
    st.audio(audio_file,format='audio/wav')
    if audio_file is not None:
        if os.path.exists("val"):
            path = os.path.join("val", audio_file.name)
            X, sample_rate = librosa.load(path)
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            feature=mfccs.reshape(1,-1)
            pred = model.predict(feature)
            if pred == ['calm']:
                st.write("I predicted the emotion as CALM")
            elif pred == ['sad']:
                st.write("I predicted the emotion as SAD")
            elif pred == ['happy']:
                st.write("I predicted the emotion as HAPPY")
            elif pred == ['angry']:
                st.write("I predicted the emotion as ANGRY")
            elif pred == ['disgust']:
                st.write("I predicted the emotion as DISGUST")
            else:
                st.write("I predicted the emotion as SURPRISE")

if __name__ == '__main__':
    main()
