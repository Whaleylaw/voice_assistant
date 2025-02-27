"""
Voice input and output functionality for the assistant.
"""

import io
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs

import config

# Initialize OpenAI client for speech recognition
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Initialize ElevenLabs client for text-to-speech
elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)


def record_audio():
    """Records audio from the microphone until Enter is pressed.
    
    Returns:
        str: The transcribed text from the audio.
    """
    audio_data = []  # List to store audio chunks
    recording = True  # Flag to control recording
    
    def record_audio_thread():
        """Continuously records audio until the recording flag is set to False."""
        nonlocal audio_data, recording
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, dtype='int16') as stream:
            print("Recording... Press Enter to stop.")
            while recording:
                audio_chunk, _ = stream.read(1024)  # Read audio data in chunks
                audio_data.append(audio_chunk)

    def stop_recording_thread():
        """Waits for user input to stop the recording."""
        input()  # Wait for Enter key press
        nonlocal recording
        recording = False

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio_thread)
    recording_thread.start()

    # Start a thread to listen for the Enter key
    stop_thread = threading.Thread(target=stop_recording_thread)
    stop_thread.start()

    # Wait for both threads to complete
    stop_thread.join()
    recording_thread.join()

    # Stack all audio chunks into a single NumPy array
    if not audio_data:
        return ""
        
    audio_data = np.concatenate(audio_data, axis=0)
    
    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    write(audio_bytes, config.SAMPLE_RATE, audio_data)
    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  # Set a filename for the in-memory file

    # Transcribe via Whisper
    try:
        transcription = openai_client.audio.transcriptions.create(
           model=config.SPEECH_TO_TEXT_MODEL, 
           file=audio_bytes,
        )
        print(f"Transcription: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


def speak(text, voice_id=None, speaking_rate=None):
    """Converts text to speech and plays it.
    
    Args:
        text (str): The text to convert to speech.
        voice_id (str, optional): The voice ID to use. Defaults to config.DEFAULT_VOICE_ID.
        speaking_rate (float, optional): The speaking rate. Defaults to config.SPEAKING_RATE.
    """
    if not text:
        return
        
    # Use default values if not specified
    voice_id = voice_id or config.DEFAULT_VOICE_ID
    speaking_rate = speaking_rate or config.SPEAKING_RATE
    
    # Prepare text by replacing ** with empty strings
    # These can cause unexpected behavior in ElevenLabs
    cleaned_text = text.replace("**", "")
    
    try:
        # Call text_to_speech API
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=cleaned_text,
            model_id=config.TEXT_TO_SPEECH_MODEL,
            voice_settings=VoiceSettings(
                stability=config.STABILITY,
                similarity_boost=config.SIMILARITY_BOOST,
                style=config.STYLE,
                use_speaker_boost=config.SPEAKER_BOOST,
            ),
        )
        
        # Play the audio
        print(f"Speaking: {cleaned_text[:50]}...")
        play(audio)
        
    except Exception as e:
        print(f"Error generating or playing speech: {e}")


def optimize_for_speech(text):
    """Optimize text for speech synthesis.
    
    This function processes text to make it more suitable for speech,
    such as expanding acronyms, fixing pronunciation issues, etc.
    
    Args:
        text (str): The text to optimize.
        
    Returns:
        str: The optimized text.
    """
    # In a more advanced implementation, we would have more sophisticated
    # processing here, but for now we'll do some basic cleanup
    
    # Replace URLs with "link"
    import re
    text = re.sub(r'https?://\S+', 'link', text)
    
    # Replace multiple newlines with single ones
    text = re.sub(r'\n\n+', '\n', text)
    
    # Replace ** markdown with emphasis in speech (for now, just remove them)
    text = text.replace('**', '')
    
    return text


if __name__ == "__main__":
    """Test the voice input and output functionality."""
    print("Testing voice input...")
    text = record_audio()
    print(f"You said: {text}")
    
    if text:
        print("Testing voice output...")
        response = f"I heard you say: {text}"
        speak(response)
