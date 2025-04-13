# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step 1: Setup Audio Recorder (ffmpeg & portaudio)
# ffmpeg, portaudio, pyaudio
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Define API Key and STT Model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
stt_model = "whisper-large-v3"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Convert speech to text using Groq API.

    Args:
    stt_model (str): The speech-to-text model.
    audio_filepath (str): Path to the audio file.
    GROQ_API_KEY (str): API key for authentication.

    Returns:
    str: Transcribed text.
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text  

def process_inputs(audio_filepath=None, text_input=None):
    """
    Process both voice and text input. If text input is provided, it is used directly.
    If text input is empty, it transcribes the voice input.

    Args:
    audio_filepath (str): Path to the audio file.
    text_input (str): Manually entered text.

    Returns:
    str: Final text from either transcription or direct input.
    """
    if text_input:  # If text input is given, use it
        logging.info("Using text input instead of voice.")
        return text_input
    elif audio_filepath:  # If no text input, use voice input
        return transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
    else:
        return "No input provided. Please provide either text or voice."
