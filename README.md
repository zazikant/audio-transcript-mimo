# Audio → Transcript | MiMo V2.5 ASR

Convert audio to text using MiMo V2.5 omnimodal ASR via OpenCode Go API.

## Features

- **Upload audio** — MP3, WAV, M4A, OGG, WEBM, FLAC, AAC
- **YouTube URL** — Download and transcribe audio from YouTube
- **Voice recording** — Record directly from your microphone, download as .wav
- **Long audio support** — Handles up to 1 hour with intelligent chunking
- **Multiple formats** — Download transcript as .txt or .json

## Setup

1. Get an API key from [opencode.ai](https://opencode.ai)
2. Enter your API key in the sidebar when you open the app

## Deploy on Streamlit Cloud

This repo is ready to deploy on Streamlit Cloud. The `packages.txt` ensures `ffmpeg` is available for audio conversion.

## Tech

- **MiMo V2.5** (OpenCode Go API) — omnimodal ASR
- **ffmpeg** — audio conversion and chunking
- **Pure-Python WAV fallback** — works without ffmpeg for WAV files
- **No pydub** — Python 3.13+ safe
