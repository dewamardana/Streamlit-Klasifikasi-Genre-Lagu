import streamlit as st
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Load the saved model and label encoder
model = load_model('best_model.h5')
label_encoder = joblib.load('labelencoder.pkl')


def split_audio(file, segment_length=3):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        trimmed_audio, _ = librosa.effects.trim(audio)
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return [], 0

    segment_frames = segment_length * sample_rate
    segments = [trimmed_audio[i:i + segment_frames]
                for i in range(0, len(trimmed_audio), segment_frames)]
    return segments, sample_rate


def ekstraksi_mfcc(audio_segment, sample_rate):
    mfcc_features = librosa.feature.mfcc(
        y=audio_segment, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    return mfcc_scaled_features


def predict_genre(file):
    segments, sample_rate = split_audio(file)
    predictions = []
    for segment in segments:
        if len(segment) == 3 * sample_rate:
            mfcc_scaled_features = ekstraksi_mfcc(segment, sample_rate)
            mfcc_scaled_features = mfcc_scaled_features.reshape(1, -1)
            predicted_label = model.predict(mfcc_scaled_features)
            predicted_label_index = np.argmax(predicted_label, axis=1)
            prediction_class = label_encoder.inverse_transform(
                predicted_label_index)
            predictions.append(prediction_class[0])
    return predictions


# UI
st.title("Music Genre Classification")
st.write(
    "Upload an audio file (in .wav format) and let's predict the genre of each segment!")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    audio_player = st.audio(uploaded_file, format='audio/wav')

    with st.spinner('Classifying...'):
        predictions = predict_genre(uploaded_file)
        st.success('Classification complete!')

    st.subheader("Predicted Genre Segments:")
    st.write(predictions)

    # Display a bar chart of the predictions
    prediction_counts = pd.Series(predictions).value_counts()
    st.subheader("Prediction Distribution:")
    st.bar_chart(prediction_counts)

    most_predicted_genre = prediction_counts.idxmax()
    st.subheader("Prediksi Genre Musik : ")
    st.write(most_predicted_genre)

    if st.checkbox("Show Model Summary"):
        st.subheader("Model Summary:")
        st.text(model.summary())
