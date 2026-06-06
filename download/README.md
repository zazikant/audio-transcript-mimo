# Audio → Knowledge Pipeline | MiMo V2.5 ASR

A Streamlit app that converts audio to structured knowledge using **MiMo V2.5** omnimodal ASR via the **OpenCode Go API**.

## Features

- 🎙️ **3 Input Modes**: Upload audio, record from mic, or paste a YouTube URL
- 🧠 **MiMo V2.5 ASR**: State-of-the-art omnimodal speech recognition
- ⏱️ **Long Audio Support**: Handles files up to 1 hour with intelligent chunking
- 🌱 **Seed Research**: Extract topics, claims, and research directions
- 🔬 **Deep Research**: Adversarial analysis to stress-test claims
- 🔄 **Circular Dependencies**: Structure and clean telegraphic notes
- 🧬 **Atomic Graph**: Knowledge graph from transcript
- 🧠 **RAG Second Brain**: Store and query documents
- 📄 **Multiple Outputs**: Download as `.txt`, `.json`, or `.md`

## Pipeline

```
🎤 Audio Input → Transcription (MiMo V2.5)
         ↓
🌱 Seed Research (extract topics & claims)
         ↓
🔬 Deep Research (adversarial analysis)
         ↓
🔄 Circular Dependencies Cleanup (structure & polish)
         ↓
🧬 Atomic Graph (knowledge map)
         ↓
🧠 RAG Second Brain (store & query)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is available (required for audio processing)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# Set your API key (optional - has a default)
export OPENCODE_API_KEY="sk-your-key-here"

# Run the app
streamlit run app.py
```

## Audio Formats Supported

MP3, WAV, M4A, OGG, WEBM, FLAC, AAC

## YouTube Downloads

YouTube requires authentication from cloud IPs. To download YouTube audio:
1. Install a browser cookie extension (e.g., "Get cookies.txt LOCALLY")
2. Export your YouTube cookies in Netscape format
3. Upload the cookies.txt file in the YouTube tab

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENCODE_API_KEY` | (built-in) | OpenCode Go API key |
| `MIMO_MODEL` | `mimo-v2.5` | Model for transcription |
| `CHUNK_DURATION_MIN` | `3` | Chunk duration in minutes |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between chunks |

## Technical Details

- **No pydub**: Uses ffmpeg directly (Python 3.13+ safe)
- **API Base URL**: `https://opencode.ai/zen/go/v1`
- **Audio Format**: `audio_url` with base64 data URI
- **Model**: `mimo-v2.5` (omnimodal, supports audio input)
