from audio_recorder_streamlit import audio_recorder
import streamlit as st
from pydub import AudioSegment
import tempfile
import io 
import speech_recognition as sr
# Record audio
def stxt_new(key, audio_bytes):
    """
    Transcribes audio bytes to text using Whisper API.

    Args:
        key (str): OpenAI API key
        audio_bytes (bytes): Audio bytes to transcribe

    Returns:
        str: Transcribed text
    """
    if audio_bytes:
        try:
            with st.spinner("Thinking..."):
                # Convert audio bytes to a WAV file
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    audio.export(temp_audio_file.name, format="wav")
                    temp_audio_filename = temp_audio_file.name

                # Use the audio file as the audio source
                r = sr.Recognizer()
                with sr.AudioFile(temp_audio_filename) as source:
                    audio = r.record(source)  # read the entire audio file

                # Recognize speech using Whisper API
                try:
                    response = r.recognize_whisper_api(audio, api_key=key)
                    return response
                except sr.RequestError as e:
                    response = "Could not request results from Whisper API"
                    st.markdown(f"<font color='red'>{response}</font>", unsafe_allow_html=True)
                except sr.UnknownValueError:
                    response = "Whisper API could not understand the audio"
                    st.markdown(f"<font color='red'>{response}</font>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<font color='red'>An error occurred: {str(e)}. The recording may be too short. We are not able to transcribe correctly your voice. Please try again</font>", unsafe_allow_html=True)
    else:
        return "No audio provided or an error occurred."
    

audio_data = audio_recorder(pause_threshold=3.0, sample_rate=48_000, icon_size="2x")
openai_api_key = st.secrets["openai"]

if audio_data:
    #st.audio(audio_data, format="audio/wav")
    prompt=stxt_new(openai_api_key,audio_data)
    st.write(prompt)
