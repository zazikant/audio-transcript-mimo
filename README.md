# Audio → Transcript | MiMo V2.5 ASR

A Streamlit app that converts audio to `transcript.txt` using **MiMo V2.5** multimodal ASR via the **OpenCode Go API**.

## Features

- 🎙️ **3 Input Modes**: Upload audio, record from mic, or paste a YouTube URL
- 🧠 **MiMo V2.5 ASR**: State-of-the-art multimodal speech recognition
- ⏱️ **Long Audio Support**: Handles files up to 1 hour with intelligent chunking
- 🔄 **Pipeline Integration**: Circular Dependencies cleanup, Atomic Graph knowledge mapping, RAG second brain
- 📄 **Multiple Outputs**: Download as `.txt` or `.json`, save to RAG

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

## Architecture

```
Input → Convert to WAV → Chunk (if >5 min) → Transcribe via MiMo V2.5 → Merge → transcript.txt
                                                                                           ↓
                                                                              Circular Dependencies Cleanup
                                                                                           ↓
                                                                              Atomic Graph Knowledge Map
                                                                                           ↓
                                                                              RAG Second Brain Storage
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENCODE_API_KEY` | (built-in) | OpenCode Go API key |
| `MIMO_MODEL` | `mimo-v2.5` | Model for transcription |
| `CHUNK_DURATION_MIN` | `3` | Chunk duration in minutes |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between chunks |

## API Details

- **Base URL**: `https://opencode.ai/zen/go/v1`
- **Model**: `mimo-v2.5` (multimodal, supports audio input)
- **Format**: OpenAI-compatible chat completions with `input_audio` content type
