# Voice Assistant with Memory

A voice-enabled personal assistant that learns about you and your business to provide better assistance over time.

## Features

- Voice input and output for natural interaction
- Persistent memory storage to remember user information and preferences
- LLM-powered responses with context from past conversations

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

4. Add your API keys to the `.env` file:
   - OPENAI_API_KEY (for Claude and Whisper)
   - ELEVENLABS_API_KEY (for voice synthesis)

## Prerequisites

- [FFmpeg](https://ffmpeg.org/) is required for the ElevenLabs voice library.
  - On macOS: `brew install ffmpeg`
  - On Windows: Download from [FFmpeg.org](https://ffmpeg.org/download.html)

## Usage

Run the test script to start the assistant:

```bash
python test_assistant.py [number_of_turns]
```

The assistant will:
1. Listen for your voice input (press Enter to stop recording)
2. Transcribe your speech to text
3. Search its memory for relevant information
4. Generate a response based on your input and memory
5. Speak the response back to you

## Project Structure

- `config.py`: Configuration settings
- `schemas.py`: Memory schema definitions
- `voice_io.py`: Voice input/output functionality
- `memory_store.py`: Memory storage and retrieval
- `agent_nodes.py`: LangGraph node implementations
- `main.py`: Graph construction and execution
- `test_assistant.py`: Test script

## Extending the Assistant

This is a foundational implementation. To extend it:

1. Add more tools in the `agent_nodes.py` file
2. Enhance memory schemas in `schemas.py`
3. Add more specialized agents for different domains
4. Implement additional voice features (wake word, continuous listening)
