#!/usr/bin/env python3
"""
Groq Whisper API — Complete Research & Implementation Reference
==============================================================

All implementation details for using Groq's Whisper API for audio transcription.

KEY FINDINGS:
  - Base URL: https://api.groq.com
  - API endpoint: POST /openai/v1/audio/transcriptions
  - Compatible with OpenAI Python SDK (just change base_url)
  - Also has its own `groq` Python SDK
  - Models: whisper-large-v3, whisper-large-v3-turbo
  - File size limit: 25 MB per request
  - Max duration: ~30 minutes per file (soft limit based on file size)
  - Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
  - Language: 99+ languages supported via ISO-639-1 codes
  - Rate limits: Varies by plan (Free: ~30 req/min, Dev: higher)
"""

import os
import json
import struct
import tempfile
import subprocess
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASE URL & API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_BASE_URL = "https://api.groq.com"
GROQ_API_ENDPOINT = "/openai/v1/audio/transcriptions"
# Full URL: https://api.groq.com/openai/v1/audio/transcriptions

# Environment variable for API key:
#   GROQ_API_KEY  (used by groq SDK)
#   Or pass api_key= explicitly


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AVAILABLE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "whisper-large-v3": {
        "description": "OpenAI Whisper large-v3, served at speed via Groq LPU",
        "accuracy": "Best accuracy, slower than turbo",
        "speed": "~30x realtime",
        "context_window": "30 seconds (Whisper native chunk size, auto-chunked internally)",
        "recommended_for": "Maximum accuracy, production use",
    },
    "whisper-large-v3-turbo": {
        "description": "Distilled/faster variant of whisper-large-v3",
        "accuracy": "Slightly lower accuracy than whisper-large-v3",
        "speed": "~50x realtime (faster)",
        "context_window": "30 seconds (same internal chunking)",
        "recommended_for": "Speed-critical applications, near-real-time",
    },
    # DEPRECATED models (do not use):
    # "whisper-large-v2" — deprecated, replaced by v3
    # "distil-whisper-large-v3-en" — English-only, deprecated
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PYTHON CODE — USING OPENAI SDK (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_with_openai_sdk(audio_path: str, api_key: str,
                                model: str = "whisper-large-v3",
                                language: str = "en",
                                prompt: str = "",
                                response_format: str = "json",
                                temperature: float = 0.0) -> dict:
    """
    Transcribe audio using Groq's Whisper API via the OpenAI Python SDK.
    
    The OpenAI SDK is fully compatible — just change base_url and api_key.
    This is the EASIEST approach if you already use the openai package.
    
    Args:
        audio_path: Path to audio file (mp3, wav, m4a, webm, etc.)
        api_key: Groq API key (from console.groq.com)
        model: "whisper-large-v3" or "whisper-large-v3-turbo"
        language: ISO-639-1 language code (e.g., "en", "es", "zh"). 
                  Optional — omit to auto-detect.
        prompt: Optional context prompt to improve accuracy (max 224 tokens)
        response_format: "json", "text", or "verbose_json"
        temperature: Sampling temperature (0.0 = deterministic)
    
    Returns:
        dict with at least 'text' key. verbose_json also includes segments/words.
    
    Example:
        >>> result = transcribe_with_openai_sdk("meeting.mp3", "gsk_...")
        >>> print(result["text"])
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",  # <-- Groq's OpenAI-compatible endpoint
        api_key=api_key,
    )

    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": response_format,
            "temperature": temperature,
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        transcription = client.audio.transcriptions.create(**kwargs)

    # Return as dict for consistency
    if response_format == "verbose_json":
        # verbose_json returns: text, segments (with start/end/words), language, duration
        return {
            "text": transcription.text,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])],
                }
                for s in (transcription.segments or [])
            ],
            "language": getattr(transcription, "language", language),
            "duration": getattr(transcription, "duration", None),
        }
    elif response_format == "text":
        return {"text": str(transcription)}
    else:
        # "json" format — returns object with .text
        return {"text": transcription.text}


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. PYTHON CODE — USING GROQ SDK (ALTERNATIVE)
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_with_groq_sdk(audio_path: str, api_key: str,
                              model: str = "whisper-large-v3",
                              language: str = "en",
                              prompt: str = "",
                              response_format: str = "json",
                              temperature: float = 0.0) -> dict:
    """
    Transcribe audio using Groq's own Python SDK.
    
    The groq SDK is a thin wrapper — functionally identical to the OpenAI SDK approach.
    Slightly more ergonomic since base_url is built-in.
    
    Install: pip install groq
    
    Args: Same as transcribe_with_openai_sdk
    
    Returns: Same format as transcribe_with_openai_sdk
    """
    from groq import Groq

    client = Groq(api_key=api_key)  # base_url defaults to https://api.groq.com

    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": response_format,
            "temperature": temperature,
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        transcription = client.audio.transcriptions.create(**kwargs)

    if response_format == "verbose_json":
        return {
            "text": transcription.text,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
                for s in (getattr(transcription, 'segments', None) or [])
            ],
            "language": getattr(transcription, "language", language),
        }
    elif response_format == "text":
        return {"text": str(transcription)}
    else:
        return {"text": transcription.text}


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. PYTHON CODE — USING RAW HTTP (curl equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_with_httpx(audio_path: str, api_key: str,
                           model: str = "whisper-large-v3",
                           language: str = "en") -> dict:
    """
    Transcribe using raw HTTP request (no SDK dependency).
    
    Useful for debugging or environments where you can't install SDKs.
    """
    import httpx

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        data = {
            "model": model,
            "response_format": "json",
            "temperature": "0.0",
        }
        if language:
            data["language"] = language

        response = httpx.post(url, headers=headers, files=files, data=data, timeout=120)

    response.raise_for_status()
    return response.json()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FILE SIZE LIMITS & LONG AUDIO HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

"""
FILE SIZE LIMIT: 25 MB per request
──────────────────────────────────

This is the hard limit. Groq will reject files > 25 MB with an error.

For reference (16kHz mono 16-bit WAV):
  - 25 MB ≈ ~13 minutes of audio
  - But MP3/OGG are ~10x smaller, so 25 MB MP3 ≈ ~2-3 hours

STRATEGY FOR LONG AUDIO:
─────────────────────────

Option A: Compress first (RECOMMENDED)
  Convert to MP3 or OGG before sending. A 1-hour WAV (~600 MB) becomes
  ~60 MB MP3 — still too large. Use 64kbps mono MP3 → ~30 MB for 1 hour.
  
Option B: Chunk the audio
  Split into segments < 25 MB each, transcribe separately, merge results.
  This is what the existing app.py does for MiMo, adapted for Groq.

Option C: Use url parameter (if available)
  The Groq SDK has a `url` parameter — you can pass a URL to an audio file
  hosted externally. This bypasses the 25 MB upload limit.
  NOTE: May not work with all file hosting services.

IMPLEMENTATION: Chunk + Merge for Groq
"""

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
CHUNK_DURATION_SEC = 120  # 2 minutes per chunk for safety with WAV


def get_audio_duration_sec(filepath: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # WAV fallback
    try:
        header = _read_wav_header(filepath)
        return header["duration_sec"]
    except Exception:
        pass

    return 0.0


def _read_wav_header(filepath: str) -> dict:
    """Read WAV file header and return metadata."""
    with open(filepath, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a WAV file")
        f.read(4)
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a WAV file")

        channels = bits_per_sample = frame_rate = num_frames = 0
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                channels = struct.unpack("<H", fmt_data[2:4])[0]
                frame_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                bytes_per_sample = bits_per_sample // 8 if bits_per_sample else 1
                num_frames = chunk_size // (channels * bytes_per_sample) if channels else 0
                break
            else:
                f.read(chunk_size)

    duration_sec = num_frames / frame_rate if frame_rate else 0
    return {
        "channels": channels,
        "frame_rate": frame_rate,
        "bits_per_sample": bits_per_sample,
        "duration_sec": duration_sec,
    }


def chunk_audio_for_groq(input_path: str, output_dir: str = None,
                          chunk_duration_sec: int = 120,
                          overlap_sec: int = 5) -> list[str]:
    """
    Split audio into chunks that fit within Groq's 25 MB limit.
    
    Strategy:
      - Convert to 16kHz mono 16-bit WAV first (standardize)
      - 16kHz mono 16-bit = 32,000 bytes/sec
      - 25 MB = 25,600,000 bytes ≈ 800 seconds ≈ 13.3 minutes
      - Use 2-minute (120s) chunks with 5s overlap for safety
      
    For MP3/OGG input, much longer chunks are possible due to compression.
    
    Args:
        input_path: Path to audio file (any format)
        output_dir: Directory for chunks (temp dir if None)
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between chunks for continuity
    
    Returns:
        List of chunk file paths
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="groq_chunks_")

    duration = get_audio_duration_sec(input_path)

    if duration <= chunk_duration_sec:
        # No chunking needed — file fits in one request
        return [input_path]

    chunks = []
    start = 0
    idx = 0

    while start < duration:
        end = min(start + chunk_duration_sec, duration)
        out_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start), "-to", str(end),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            "-f", "wav", out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(out_path):
            chunks.append(out_path)

        if end >= duration:
            break
        start = end - overlap_sec  # Overlap for continuity
        idx += 1

    return chunks


def transcribe_long_audio(audio_path: str, api_key: str,
                           model: str = "whisper-large-v3",
                           language: str = "en",
                           chunk_duration_sec: int = 120,
                           overlap_sec: int = 5) -> str:
    """
    Transcribe audio of any length by chunking and merging.
    
    Handles the 25 MB file size limit by splitting audio into manageable
    segments, transcribing each, and merging with overlap deduplication.
    
    This is the PRODUCTION-READY approach for long audio with Groq.
    """
    from groq import Groq

    client = Groq(api_key=api_key)

    # Step 1: Chunk the audio
    chunks = chunk_audio_for_groq(
        audio_path,
        chunk_duration_sec=chunk_duration_sec,
        overlap_sec=overlap_sec
    )

    if len(chunks) == 1:
        # Simple case — one API call
        with open(chunks[0], "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language if language != "auto" else None,
                response_format="json",
                temperature=0.0,
            )
        return transcription.text

    # Step 2: Transcribe each chunk with context from previous
    transcript_parts = []
    for i, chunk_path in enumerate(chunks):
        prompt_context = " ".join(transcript_parts[-1:])[-224:] if transcript_parts else ""

        with open(chunk_path, "rb") as f:
            try:
                transcription = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    language=language if language != "auto" else None,
                    prompt=prompt_context,  # Help maintain continuity
                    response_format="json",
                    temperature=0.0,
                )
                part_text = transcription.text
                if part_text:
                    transcript_parts.append(part_text)
            except Exception as e:
                print(f"Error transcribing chunk {i}: {e}")
                transcript_parts.append(f"[ERROR chunk {i}: {e}]")

    # Step 3: Merge with overlap deduplication
    return _merge_transcript_parts(transcript_parts)


def _merge_transcript_parts(parts: list[str]) -> str:
    """Merge transcript chunks, deduplicating overlap regions."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    merged = parts[0]
    for i in range(1, len(parts)):
        next_part = parts[i]
        overlap_found = False
        merged_words = merged.split()
        next_words = next_part.split()

        check_len = min(30, len(merged_words), len(next_words))
        if check_len > 3:
            for window in range(check_len, 2, -1):
                merged_tail = " ".join(merged_words[-window:]).lower().strip(".,!?;:")
                next_head = " ".join(next_words[:window]).lower().strip(".,!?;:")
                if merged_tail == next_head and len(merged_tail) > 10:
                    merged = merged + " " + " ".join(next_words[window:])
                    overlap_found = True
                    break

        if not overlap_found:
            merged = merged + "\n\n" + next_part

    return merged.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUPPORTED AUDIO FORMATS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_FORMATS = {
    "mp3":  {"mime": "audio/mpeg",    "extension": ".mp3",  "notes": "Most compact, recommended for long audio"},
    "mp4":  {"mime": "audio/mp4",     "extension": ".mp4",  "notes": "M4A/AAC container"},
    "mpeg": {"mime": "audio/mpeg",    "extension": ".mpeg", "notes": "MPEG audio"},
    "mpga": {"mime": "audio/mpeg",    "extension": ".mpga", "notes": "MPEG audio"},
    "m4a":  {"mime": "audio/m4a",     "extension": ".m4a",  "notes": "AAC audio, good compression"},
    "wav":  {"mime": "audio/wav",     "extension": ".wav",  "notes": "Uncompressed, large files. 25MB ≈ 13min at 16kHz mono"},
    "webm": {"mime": "audio/webm",    "extension": ".webm", "notes": "WebM/Opus, common from browser recording"},
}

# IMPORTANT: Groq does NOT natively support FLAC, OGG (Vorbis), or AIFF.
# If you have these formats, convert to MP3/WAV/M4A first using ffmpeg:
#   ffmpeg -i input.flac -ar 16000 -ac 1 output.mp3


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LANGUAGE PARAMETER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

"""
Groq Whisper supports 99+ languages via ISO-639-1 codes.

The `language` parameter is OPTIONAL — if omitted, the model auto-detects the language.
Setting it explicitly can improve accuracy for the target language.

FULL LIST OF SUPPORTED LANGUAGES (from SDK type hints):
  en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi,
  fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la,
  mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy,
  ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be,
  tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl,
  mg, as, tt, haw, ln, ha, ba, jv, su, yue

NOTES:
  - "yue" = Cantonese Chinese
  - "zh" = Mandarin Chinese
  - Language parameter improves accuracy but is NOT required
  - Auto-detection works well for most major languages
  - For code-switching (mixed languages), omit language parameter
"""

SUPPORTED_LANGUAGES = {
    "en": "English", "zh": "Chinese (Mandarin)", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Norwegian Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese",
    "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jv": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RATE LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Rate limits vary by plan and are subject to change. As of 2025:

FREE TIER:
  - Requests: 30 requests per minute (approximate)
  - Tokens: 7,500 tokens per minute for audio
  - Daily limit: May apply based on usage patterns

DEVELOPER TIER (pay-as-you-go):
  - Requests: Higher limits (varies, typically 60-100 req/min)
  - Tokens: Significantly higher token limits
  - No hard daily limit (usage-based billing)

ENTERPRISE:
  - Custom limits based on agreement

IMPORTANT NOTES:
  - Rate limits are per API key, not per model
  - Both whisper-large-v3 and whisper-large-v3-turbo share the same limits
  - Audio transcription consumes tokens proportional to audio duration
  - 1 minute of audio ≈ ~150 tokens (approximate)
  - Exceeding rate limits returns HTTP 429 with Retry-After header
  
RETRY STRATEGY:
  - Exponential backoff: 1s, 2s, 4s, 8s
  - Respect Retry-After header from 429 responses
  - Both groq and openai SDKs handle retries automatically
"""

RATE_LIMITS = {
    "free": {
        "requests_per_minute": 30,
        "tokens_per_minute": 7500,
        "max_file_size_mb": 25,
    },
    "developer": {
        "requests_per_minute": 100,  # approximate
        "tokens_per_minute": 30000,  # approximate
        "max_file_size_mb": 25,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RESPONSE FORMATS
# ═══════════════════════════════════════════════════════════════════════════════

"""
response_format="json" (default):
  Returns: {"text": "transcribed text here"}

response_format="text":
  Returns: plain text string of the transcription

response_format="verbose_json":
  Returns rich object with:
  {
    "text": "full transcription",
    "segments": [
      {"id": 0, "start": 0.0, "end": 3.5, "text": "Hello world"},
      {"id": 1, "start": 3.5, "end": 7.2, "text": "How are you"},
    ],
    "language": "en",
    "duration": 7.2,
    "words": [...]  # if timestamp_granularities=["word"] is set
  }

TIMESTAMP GRANULARITIES:
  - timestamp_granularities=["segment"] — segment-level timestamps (default)
  - timestamp_granularities=["word"] — word-level timestamps
  - timestamp_granularities=["word", "segment"] — both word and segment timestamps
  - Only works with response_format="verbose_json"
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PROMPT PARAMETER
# ═══════════════════════════════════════════════════════════════════════════════

"""
The `prompt` parameter improves transcription accuracy by providing context.

Usage:
  - Max 224 tokens
  - Should be in the same language as the audio
  - Can include vocabulary, names, acronyms, or previous transcript text
  - The model uses this as a hint, not a strict template

Examples:
  prompt="The following is a meeting about quarterly earnings."
  prompt="Speakers: Dr. Sarah Chen, Prof. James Williams"
  prompt="Technical terms: Kubernetes, PostgreSQL, microservices"

For chunked transcription, pass the last ~224 tokens of the previous
transcription as the prompt to maintain continuity.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 10. URL PARAMETER (REMOTE AUDIO)
# ═══════════════════════════════════════════════════════════════════════════════

"""
The Groq SDK supports a `url` parameter as an alternative to `file`:

    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        url="https://example.com/audio.mp3",
        response_format="json",
    )

This allows you to pass a URL to a hosted audio file instead of uploading.
Potentially bypasses the 25MB upload limit since the file is fetched server-side.

NOTE: The URL must be publicly accessible. Signed/expiring URLs may work.
This parameter may not be documented in all API references.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMPARISON: GROQ WHISPER vs CURRENT MIMO V2.5
# ═══════════════════════════════════════════════════════════════════════════════

COMPARISON = {
    "groq_whisper": {
        "approach": "Dedicated ASR model (Whisper)",
        "api_style": "OpenAI-compatible REST API",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["whisper-large-v3", "whisper-large-v3-turbo"],
        "purpose_built": True,  # Purpose-built for speech-to-text
        "speed": "30-50x realtime",
        "accuracy": "Excellent for clean speech",
        "file_size_limit": "25 MB",
        "chunking": "Internal (30s windows) + external for long audio",
        "languages": "99+ languages",
        "timestamps": "Yes (segment and word-level via verbose_json)",
        "cost": "Free tier available, pay-as-you-go for higher limits",
        "output_format": "Plain text, JSON, verbose JSON with segments/words",
        "streaming": "No (batch only)",
    },
    "mimo_v25": {
        "approach": "Omnimodal LLM (chat completions with audio)",
        "api_style": "OpenAI chat completions API with audio_url",
        "base_url": "https://opencode.ai/zen/go/v1",
        "models": ["mimo-v2-omni", "mimo-v2.5"],
        "purpose_built": False,  # General multimodal model, not ASR-specific
        "speed": "Variable (LLM inference speed)",
        "accuracy": "Good, but can hallucinate or add commentary",
        "file_size_limit": "Varies (base64 encoded in request body)",
        "chunking": "Manual chunking required (app.py does this)",
        "languages": "Multi-language (via LLM capability)",
        "timestamps": "No native timestamp support",
        "cost": "Per-token LLM pricing (more expensive)",
        "output_format": "Free-form text (LLM output, may need filtering)",
        "streaming": "Possible (chat completions streaming)",
    },
}

"""
KEY ADVANTAGES OF SWITCHING TO GROQ WHISPER:
1. Purpose-built for ASR — no hallucination or commentary in output
2. Structured output with timestamps (verbose_json)
3. 30-50x realtime speed on Groq LPUs
4. Free tier available
5. Clean API — no need for prompt engineering or output filtering
6. Word-level timestamps for subtitle/caption generation
7. 99+ language support with explicit language codes

KEY ADVANTAGES OF STAYING WITH MIMO:
1. Can understand context and handle complex audio (multiple speakers, music+speech)
2. Can follow custom instructions in system prompt
3. Streaming support for real-time applications
4. May handle more audio formats natively
5. Can do more than just transcription (summarization, etc. in same call)

RECOMMENDATION: Use Groq Whisper for pure transcription tasks.
Use MiMo for tasks requiring understanding beyond raw transcription.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 12. QUICK-START EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def quick_start():
    """
    Minimal example to transcribe an audio file with Groq Whisper.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python groq_whisper_research.py <audio_file> [api_key]")
        print()
        print("Set GROQ_API_KEY environment variable or pass as 2nd argument.")
        return

    audio_file = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("ERROR: No API key provided. Set GROQ_API_KEY or pass as argument.")
        return

    if not os.path.exists(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        return

    # Simple transcription
    print(f"Transcribing: {audio_file}")
    print(f"Using model:  whisper-large-v3")

    result = transcribe_with_groq_sdk(
        audio_path=audio_file,
        api_key=api_key,
        model="whisper-large-v3",
        language="en",
        response_format="verbose_json",
    )

    print()
    print("=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(result["text"])

    if result.get("segments"):
        print()
        print("=" * 60)
        print("SEGMENTS:")
        print("=" * 60)
        for seg in result["segments"]:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            print(f"  [{start:6.1f}s - {end:6.1f}s] {text}")


if __name__ == "__main__":
    quick_start()
