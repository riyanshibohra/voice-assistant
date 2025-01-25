# VoiceGPT 🎙️ - Voice Assistant

A sophisticated voice assistant that combines speech recognition, natural language processing, and knowledge base capabilities. Built with OpenAI's Whisper, GPT, ElevenLabs, and Pinecone for an interactive and intelligent conversational experience.

> This project is adapted from [JarvisBase](https://github.com/peterw/JarvisBase) by Peter W. The original project uses DeepLake for vector storage, while this adaptation uses Pinecone.

## Features

- 🎤 **Voice Input**: Record and transcribe voice using OpenAI's Whisper API
- 💬 **Text Chat**: Fallback text input option for interactions
- 🔍 **Knowledge Base**: Integrated with Hugging Face documentation for intelligent responses
- 🗣️ **Voice Output**: Text-to-speech responses using ElevenLabs
- 🧠 **Vector Search**: Efficient document retrieval using Pinecone
- 🌐 **Web Scraping**: Automated documentation scraping from Hugging Face

## Prerequisites

- Python 3.11+
- OpenAI API key
- ElevenLabs API key
- Pinecone API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/voice-assistant.git
cd voice-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key
ELEVEN_API_KEY=your_elevenlabs_key
PINECONE_API_KEY=your_pinecone_key
```

## Usage

1. Initialize the knowledge base:
```bash
python scrape.py
```

2. Start the voice assistant:
```bash
streamlit run chat.py
```

3. Use the interface to:
   - Record voice input
   - View transcriptions
   - Get AI responses
   - Listen to voice responses

## Project Structure

- `chat.py`: Main application with Streamlit interface
- `scrape.py`: Knowledge base initialization and web scraping
- `requirements.txt`: Project dependencies
- `.env`: API key configuration
- `init.ipynb`: Development notebook with implementation details

## Dependencies

- langchain: Framework for developing applications powered by language models
- openai: OpenAI API client for GPT and Whisper
- elevenlabs: Text-to-speech capabilities
- streamlit: Web interface
- pinecone-client: Vector database for efficient similarity search
- beautifulsoup4: Web scraping
- audio-recorder-streamlit: Voice recording component