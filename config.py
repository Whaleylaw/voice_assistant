"""
Configuration settings for the voice assistant.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Models
SPEECH_TO_TEXT_MODEL = "whisper-1"
CHAT_MODEL = "anthropic:claude-3-5-sonnet-latest"
TEXT_TO_SPEECH_MODEL = "eleven_turbo_v2_5"
EMBEDDING_MODEL = "openai:text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Voice settings
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # ElevenLabs Adam voice
SPEAKING_RATE = 1.0
STABILITY = 0.0
SIMILARITY_BOOST = 1.0
STYLE = 0.0
SPEAKER_BOOST = True

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Memory settings
DEFAULT_USER_ID = "default-user"
MEMORY_NAMESPACE_USER_PROFILE = "user_profile"
MEMORY_NAMESPACE_BUSINESS = "business_knowledge"
MEMORY_NAMESPACE_CONVERSATION = "conversation_history"
MEMORY_NAMESPACE_INSTRUCTIONS = "instructions"

# Conversation settings
MAX_CONVERSATION_HISTORY = 10  # Number of recent messages to keep in context
