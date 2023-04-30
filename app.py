import streamlit as st
import streamlit_audio_recorder as st_audiorec

# Add an instance of the audio recorder component to your streamlit app's code.
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # display audio data as received on the backend
    st.audio(wav_audio_data, format='audio/wav')
