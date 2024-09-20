import streamlit as st
from gtts import gTTS
import tempfile
import os

import base64


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


def tts(text,language):
    # Create a temporary file to save the audio
    languages_dict={
        "English":"en",
        "French":"fr"
    }
    language_code=languages_dict[language]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_filepath = temp_audio.name

        # Generate gTTS audio and save it to the temporary file
        tts_response = gTTS(text, lang=language_code, slow=False)
        tts_response.save(temp_filepath)

        # Display the audio using st.audio
        #autoplay_audio(temp_filepath)
        st.audio(temp_filepath, format="audio/mp3", start_time=0)

    # Delete the temporary file after displaying
    os.remove(temp_filepath)
