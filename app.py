"""
Audio → Transcript — MiMo V2.5 / Groq Whisper
===============================================
Converts audio (uploaded, recorded, or from YouTube) to transcript
using MiMo V2.5 ASR (via OpenCode Go API) or Groq Whisper.

Pipeline:
  1. Audio input (upload / YouTube / record)
  2. Convert to 16 kHz mono WAV (ffmpeg or pure-Python fallback)
  3. Transcribe via selected provider (MiMo V2.5 or Groq Whisper)
  4. Display editable transcript with download / copy

API keys persist in URL query params so they survive page reload.
"""

import os
import json
import time
import base64
import shutil
import tempfile
import logging
import struct
import subprocess
import html as html_mod
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI


# ─── Configuration ────────────────────────────────────────────────────────────

OPCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
MIMO_MODEL = "mimo-v2.5"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Audio chunking config
CHUNK_DURATION_MIN = int(os.environ.get("CHUNK_DURATION_MIN", "3"))
CHUNK_OVERLAP_SEC = int(os.environ.get("CHUNK_OVERLAP_SEC", "5"))
MAX_AUDIO_DURATION = 60  # minutes (1 hour)

# Output
TRANSCRIPT_FILENAME = "transcript.txt"

# Large-doc thresholds (borrowed from PDF-to-MD pattern)
LARGE_DOC_CHARS = 50_000
TRUNCATE_PREVIEW_CHARS = 5_000
PREVIEW_HEIGHT_PX = 480

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── Clipboard Helper (from PDF-to-MD pattern) ───────────────────────────────

def copy_button(text: str, label: str = "Copy to Clipboard", key_suffix: str = ""):
    """Clipboard copy that works for arbitrarily large outputs (32k+).

    Key technique: the full text is written into a hidden <textarea> via
    html.escape() at render time. The JS reads from that DOM node — not a JS
    string literal — so there is no browser template-literal size cap or
    Streamlit iframe serialization truncation regardless of output length.

    Falls back from navigator.clipboard.writeText() to
    document.execCommand('copy') for older browsers.
    """
    encoded = html_mod.escape(text, quote=True)
    uid = abs(hash(text + key_suffix)) % 1_000_000
    btn_id = f"copy-btn-{uid}"
    ta_id = f"copy-ta-{uid}"

    components.html(
        f"""
        <textarea id="{ta_id}"
            style="position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;"
        >{encoded}</textarea>
        <button id="{btn_id}"
            style="background:#1f77b4;color:white;border:none;padding:10px 20px;
                   border-radius:6px;font-size:14px;cursor:pointer;width:100%;margin-top:4px;">
            {label}
        </button>
        <script>
        (function() {{
            var btn = document.getElementById('{btn_id}');
            var ta  = document.getElementById('{ta_id}');
            btn.addEventListener('click', function() {{
                var txt = ta.value;
                function markDone() {{
                    btn.innerText = 'Copied!';
                    btn.style.background = '#2d6a2d';
                    setTimeout(function() {{
                        btn.innerText = '{label}';
                        btn.style.background = '#1f77b4';
                    }}, 2000);
                }}
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(txt).then(markDone).catch(function() {{
                        ta.style.cssText = 'position:static;width:100%;height:2px;';
                        ta.select();
                        document.execCommand('copy');
                        ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                        markDone();
                    }});
                }} else {{
                    ta.style.cssText = 'position:static;width:100%;height:2px;';
                    ta.select();
                    document.execCommand('copy');
                    ta.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;';
                    markDone();
                }}
            }});
        }})();
        </script>
        """,
        height=60,
    )


# ─── Scrollable Preview Container (from PDF-to-MD pattern) ───────────────────

def safe_transcript_display(text: str, view_mode: str = "Rendered"):
    """Display transcript output inside a fixed-height scrollable container.

    The container has its own scrollbar so the page doesn't grow endlessly.
    Download / copy buttons live OUTSIDE this container so they are always
    visible without scrolling.

    For very large transcripts (> LARGE_DOC_CHARS), rendered view is truncated
    to avoid rendering 100k+ DOM nodes inside the container.
    """
    char_count = len(text)
    is_large = char_count > LARGE_DOC_CHARS

    with st.container(height=PREVIEW_HEIGHT_PX):
        if is_large:
            st.warning(
                f"Large transcript ({char_count:,} chars). "
                "Preview truncated for performance. Use Download or Copy for full output."
            )

        if view_mode == "Rendered":
            if is_large:
                preview = text[:TRUNCATE_PREVIEW_CHARS]
                st.markdown(
                    f"{preview}\n\n> ... *(truncated — {char_count - TRUNCATE_PREVIEW_CHARS:,} more chars)*"
                )
            else:
                st.markdown(text)
        else:  # Raw text view
            st.code(text, language="text")


# ─── API Clients ──────────────────────────────────────────────────────────────

def get_opencode_client(api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at OpenCode Go API (MiMo V2.5)."""
    return OpenAI(
        base_url=OPCODE_BASE_URL,
        api_key=api_key,
    )


def get_groq_client(api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at Groq API (Whisper)."""
    return OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=api_key,
    )


# ─── Audio Utilities (pydub-free, Python 3.13+ safe) ────────────────────────

def _ffmpeg_available() -> bool:
    """Check if ffmpeg/ffprobe are available on the system."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_FFMPEG_OK = None  # Lazy-checked


def _check_ffmpeg() -> bool:
    """Lazy check for ffmpeg availability, cached."""
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        _FFMPEG_OK = _ffmpeg_available()
        if not _FFMPEG_OK:
            log.warning("ffmpeg not found — using pure-Python WAV fallback (non-WAV formats will fail)")
    return _FFMPEG_OK


# ── Pure-Python WAV helpers (no ffmpeg needed) ─────────────────────────────

def _read_wav_header(filepath: str) -> dict:
    """Read WAV file header and return metadata dict."""
    with open(filepath, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a WAV file (RIFF header missing): {riff}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAV file (WAVE marker missing): {wave}")

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
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id == b"data":
                bytes_per_sample = bits_per_sample // 8 if bits_per_sample else 1
                num_frames = chunk_size // (channels * bytes_per_sample) if channels and bits_per_sample else 0
                break
            else:
                f.read(chunk_size)

    duration_sec = num_frames / frame_rate if frame_rate else 0
    return {
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "bytes_per_sample": bits_per_sample // 8 if bits_per_sample else 1,
        "frame_rate": frame_rate,
        "num_frames": num_frames,
        "duration_sec": duration_sec,
    }


def _resample_wav(input_path: str, output_path: str,
                  target_rate: int = 16000, target_channels: int = 1) -> str:
    """Convert WAV to target sample rate and channels using pure Python."""
    header = _read_wav_header(input_path)
    src_rate = header["frame_rate"]
    src_channels = header["channels"]
    src_bits = header["bits_per_sample"]
    src_bytes = header["bytes_per_sample"]

    raw_data = b""
    with open(input_path, "rb") as f:
        f.read(12)
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size_bytes = f.read(4)
            if len(chunk_size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
            if chunk_id == b"data":
                raw_data = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)

    if not raw_data:
        raise ValueError(f"Could not find data chunk in WAV file: {input_path}")

    total_samples = len(raw_data) // src_bytes

    if src_bits == 16:
        fmt = "<" + "h" * total_samples
        samples = list(struct.unpack(fmt, raw_data[:total_samples * 2]))
    elif src_bits == 8:
        samples = [((b - 128) / 128.0) * 32767 for b in raw_data]
    elif src_bits == 32:
        fmt = "<" + "i" * total_samples
        samples = [s >> 16 for s in struct.unpack(fmt, raw_data[:total_samples * 4])]
    else:
        samples = list(struct.unpack("<" + "h" * (len(raw_data) // 2), raw_data[:len(raw_data) // 2 * 2]))

    if src_channels > 1:
        mono_samples = []
        for i in range(0, len(samples), src_channels):
            frame = samples[i:i + src_channels]
            mono_samples.append(sum(frame) // src_channels)
        samples = mono_samples

    if src_rate != target_rate:
        ratio = src_rate / target_rate
        new_length = int(len(samples) / ratio)
        resampled = []
        for i in range(new_length):
            src_pos = i * ratio
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(samples):
                val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                val = samples[idx] if idx < len(samples) else 0
            resampled.append(max(-32768, min(32767, val)))
        samples = resampled

    num_frames = len(samples)
    data_size = num_frames * 2
    clamped = [max(-32768, min(32767, s)) for s in samples]
    audio_data = struct.pack("<" + "h" * num_frames, *clamped)

    with open(output_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", target_rate))
        f.write(struct.pack("<I", target_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_data)

    return output_path


def _chunk_wav_python(input_wav: str, chunk_minutes: int, overlap_seconds: int,
                      output_dir: str) -> list[str]:
    """Split a WAV file into chunks using pure Python (no ffmpeg)."""
    header = _read_wav_header(input_wav)
    frame_rate = header["frame_rate"]
    channels = header["channels"]
    bits_per_sample = header["bits_per_sample"]
    bytes_per_sample = header["bytes_per_sample"]

    chunk_sec = chunk_minutes * 60
    overlap_sec = overlap_seconds
    total_sec = header["duration_sec"]

    if total_sec <= chunk_sec:
        return [input_wav]

    raw_data = b""
    with open(input_wav, "rb") as f:
        f.read(12)
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size_bytes = f.read(4)
            if len(chunk_size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
            if chunk_id == b"data":
                raw_data = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)

    if not raw_data:
        raise ValueError(f"Could not find data chunk in WAV file: {input_wav}")

    bytes_per_frame = channels * bytes_per_sample
    total_frames = len(raw_data) // bytes_per_frame
    chunk_frames = int(chunk_sec * frame_rate)
    overlap_frames = int(overlap_sec * frame_rate)

    chunks = []
    start_frame = 0
    idx = 0

    while start_frame < total_frames:
        end_frame = min(start_frame + chunk_frames, total_frames)
        start_byte = start_frame * bytes_per_frame
        end_byte = end_frame * bytes_per_frame
        chunk_data = raw_data[start_byte:end_byte]
        out_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        data_size = len(chunk_data)

        with open(out_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", channels))
            f.write(struct.pack("<I", frame_rate))
            f.write(struct.pack("<I", frame_rate * bytes_per_frame))
            f.write(struct.pack("<H", bytes_per_frame))
            f.write(struct.pack("<H", bits_per_sample))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(chunk_data)

        chunks.append(out_path)
        if end_frame >= total_frames:
            break
        start_frame = end_frame - overlap_frames
        idx += 1

    return chunks


# ── High-level audio API ────────────────────────────────────────────────────

def get_audio_duration(filepath: str) -> float:
    """Return audio duration in minutes."""
    if _check_ffmpeg():
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "N/A":
            try:
                return float(result.stdout.strip()) / 60.0
            except ValueError:
                pass

    header = _read_wav_header(filepath)
    return header["duration_sec"] / 60.0


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """Convert any audio file to 16kHz mono 16-bit PCM WAV."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    if _check_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            "-f", "wav", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return output_path
        log.warning(f"ffmpeg conversion failed, trying WAV fallback: {result.stderr[-200:]}")

    try:
        header = _read_wav_header(input_path)
        if header["frame_rate"] == 16000 and header["channels"] == 1 and header["bits_per_sample"] == 16:
            shutil.copy2(input_path, output_path)
            return output_path
        return _resample_wav(input_path, output_path)
    except (ValueError, struct.error) as e:
        raise RuntimeError(
            f"Cannot convert {input_path}: ffmpeg is not installed and the file is not a valid WAV. "
            f"Install ffmpeg or upload a WAV file. Error: {e}"
        )


def chunk_audio_wav(input_wav: str, chunk_minutes: int = CHUNK_DURATION_MIN,
                    overlap_seconds: int = CHUNK_OVERLAP_SEC,
                    output_dir: str = None) -> list[str]:
    """Split a WAV file into overlapping chunks."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")

    total_duration_min = get_audio_duration(input_wav)
    chunk_sec = chunk_minutes * 60
    total_sec = total_duration_min * 60

    if total_sec <= chunk_sec:
        return [input_wav]

    if _check_ffmpeg():
        overlap_sec = overlap_seconds
        chunks = []
        start = 0
        idx = 0
        while start < total_sec:
            end = min(start + chunk_sec, total_sec)
            out_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
            cmd = [
                "ffmpeg", "-y", "-i", input_wav,
                "-ss", str(start), "-to", str(end),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                "-f", "wav", out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                chunks.append(out_path)
            if end >= total_sec:
                break
            start = end - overlap_sec
            idx += 1
        return chunks

    return _chunk_wav_python(input_wav, chunk_minutes, overlap_seconds, output_dir)


def wav_to_base64(filepath: str) -> str:
    """Read a WAV file and return base64-encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_uploaded_file(uploaded_file, output_dir: str = None) -> str:
    """Save a Streamlit UploadedFile to disk and return its path.

    Detects the actual audio format from magic bytes and corrects the file
    extension so ffmpeg and other tools can process it correctly.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="upload_")

    original_suffix = Path(uploaded_file.name).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=original_suffix, dir=output_dir)
    os.close(fd)

    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    actual_format = detect_audio_format(path)
    expected_extensions = {
        "wav": ".wav", "mp3": ".mp3", "ogg": ".ogg",
        "flac": ".flac", "webm": ".webm", "m4a": ".m4a",
    }

    if actual_format in expected_extensions:
        correct_suffix = expected_extensions[actual_format]
        if original_suffix.lower() != correct_suffix:
            new_path = path.rsplit(".", 1)[0] + correct_suffix
            os.rename(path, new_path)
            log.info(f"Corrected file extension: {original_suffix} → {correct_suffix} (detected {actual_format})")
            return new_path

    return path


def detect_audio_format(filepath: str) -> str:
    """Detect the actual audio format by reading file magic bytes."""
    with open(filepath, "rb") as f:
        header = f.read(16)

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    elif header[:4] == b"OggS":
        return "ogg"
    elif header[:3] == b"ID3" or header[:2] == b"\xff\xfb" or header[:2] == b"\xff\xf3":
        return "mp3"
    elif header[:4] == b"fLaC":
        return "flac"
    elif header[:4] == b"\x1a\x45\xdf\xa5":
        return "webm"
    elif header[4:8] == b"ftyp":
        return "m4a"
    else:
        return "unknown"


# ─── Transcription: MiMo V2.5 ───────────────────────────────────────────────

def transcribe_chunk_mimo(wav_path: str, api_key: str,
                          language: str = "en", prompt_context: str = "") -> str:
    """Transcribe a single audio chunk using MiMo V2.5 via OpenCode Go API."""
    client = get_opencode_client(api_key)

    audio_b64 = wav_to_base64(wav_path)
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    system_msg = (
        "You are a professional speech-to-text transcription engine. "
        "Transcribe the provided audio accurately and faithfully. "
        "Output ONLY the transcript text — no timestamps, no commentary, no labels. "
        "Preserve speaker turns with newlines. "
        "If the audio is not speech (music, noise, silence), respond with: [non-speech]"
    )

    user_parts = [
        {"type": "text", "text": f"Transcribe this audio. Language: {language}."},
    ]
    if prompt_context:
        user_parts[0]["text"] += (
            f"\n\nContext from previous chunk (for continuity):\n{prompt_context[-500:]}"
        )

    user_parts.append({
        "type": "audio_url",
        "audio_url": {"url": audio_data_uri}
    })

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_parts},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            msg = response.choices[0].message
            result = None

            if msg.content:
                result = msg.content.strip()

            # Fallback: reasoning_content (MiMo puts output here sometimes)
            if not result:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    lines = rc.strip().split("\n")
                    transcript_lines = []
                    skip_patterns = ["The user", "Let me", "I need to", "This is",
                                     "I'll", "Looking at", "Checking", "Analyzing",
                                     "The audio", "I can hear", "I should"]
                    for line in lines:
                        stripped = line.strip()
                        if stripped and not any(stripped.startswith(p) for p in skip_patterns):
                            if stripped.startswith('"') or stripped.startswith("'"):
                                cleaned = stripped.strip('"').strip("'")
                                if len(cleaned) > 5:
                                    transcript_lines.append(cleaned)
                            elif len(stripped) > 20 and not stripped.startswith("Wait"):
                                transcript_lines.append(stripped)

                    if transcript_lines:
                        result = " ".join(transcript_lines)
                    else:
                        result = rc.strip()

            if not result:
                if attempt < max_retries - 1:
                    log.warning(f"Empty API response, retry {attempt+1}/{max_retries}")
                    time.sleep(2 * (attempt + 1))
                    continue
                return "[TRANSCRIPTION_ERROR: Empty response from API]"

            return result

        except Exception as e:
            error_str = str(e)
            log.error(f"MiMo API error (attempt {attempt+1}): {e}")
            if "image input" in error_str.lower() or "no endpoints found" in error_str.lower():
                return "[TRANSCRIPTION_ERROR: MiMo V2.5 does not support audio input on this endpoint.]"
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return f"[TRANSCRIPTION_ERROR: {e}]"


# ─── Transcription: Groq Whisper ─────────────────────────────────────────────

def transcribe_chunk_groq(wav_path: str, api_key: str,
                          language: str = "en", prompt_context: str = "") -> str:
    """Transcribe a single audio chunk using Groq Whisper API."""
    client = get_groq_client(api_key)

    with open(wav_path, "rb") as audio_file:
        kwargs = {
            "model": GROQ_WHISPER_MODEL,
            "file": audio_file,
            "response_format": "text",
        }
        # Only pass language if it's not "auto"
        if language and language != "auto":
            kwargs["language"] = language
        # prompt_context gives Whisper continuity between chunks
        if prompt_context:
            kwargs["prompt"] = prompt_context[-500:]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                transcript = client.audio.transcriptions.create(**kwargs)
                result = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
                if not result:
                    if attempt < max_retries - 1:
                        log.warning(f"Empty Groq response, retry {attempt+1}/{max_retries}")
                        time.sleep(2 * (attempt + 1))
                        continue
                    return "[TRANSCRIPTION_ERROR: Empty response from Groq API]"
                return result
            except Exception as e:
                error_str = str(e)
                log.error(f"Groq API error (attempt {attempt+1}): {e}")
                if "invalid api key" in error_str.lower() or "auth" in error_str.lower():
                    return "[TRANSCRIPTION_ERROR: Invalid Groq API key. Check your key in the sidebar.]"
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    return f"[TRANSCRIPTION_ERROR: {e}]"


# ─── Unified Transcription (auto-chunking) ───────────────────────────────────

def transcribe_audio(wav_path: str, provider: str, api_key: str,
                     language: str = "en", progress_callback=None) -> str:
    """Transcribe a full audio file (handles chunking for long audio)."""
    duration_min = get_audio_duration(wav_path)
    log.info(f"Transcribing audio: {duration_min:.1f} minutes via {provider}")

    transcribe_fn = transcribe_chunk_groq if provider == "groq" else transcribe_chunk_mimo

    if duration_min <= 5:
        if progress_callback:
            progress_callback(0.1, "Transcribing audio...")
        transcript = transcribe_fn(wav_path, api_key=api_key, language=language)
        if progress_callback:
            progress_callback(1.0, "Done!")
        return transcript

    # Long audio: chunk with overlap
    if duration_min <= 15:
        chunk_min = 5
    elif duration_min <= 30:
        chunk_min = 4
    else:
        chunk_min = 3

    chunks = chunk_audio_wav(wav_path, chunk_minutes=chunk_min)
    log.info(f"Split into {len(chunks)} chunks of ~{chunk_min} min")

    transcript_parts = []
    for i, chunk_path in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        msg = f"Transcribing chunk {i+1}/{len(chunks)}..."
        if progress_callback:
            progress_callback(progress * 0.9, msg)

        context = " ".join(transcript_parts[-2:]) if transcript_parts else ""
        part = transcribe_fn(chunk_path, api_key=api_key,
                             language=language, prompt_context=context)

        if part and part != "[non-speech]":
            transcript_parts.append(part)

        if i < len(chunks) - 1:
            time.sleep(1)

    full_transcript = merge_transcript_parts(transcript_parts)

    if progress_callback:
        progress_callback(1.0, "Transcription complete!")

    return full_transcript


def merge_transcript_parts(parts: list[str]) -> str:
    """Merge transcript chunks, handling overlap deduplication."""
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


# ─── YouTube Download ────────────────────────────────────────────────────────

def download_youtube_audio(url: str, cookies_path: str = None) -> tuple[str, str]:
    """Download audio from a YouTube URL using yt-dlp."""
    output_dir = tempfile.mkdtemp(prefix="yt_download_")
    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--print", "title",
    ]

    if cookies_path:
        cmd.extend(["--cookies", cookies_path])

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            error_msg = result.stderr[-500:]
            if "Sign in to confirm" in error_msg or "bot" in error_msg:
                raise RuntimeError(
                    "YouTube requires authentication. Please upload your browser cookies "
                    "file (Netscape format) using the 'Upload Cookies' button below."
                )
            raise RuntimeError(f"yt-dlp failed: {error_msg}")

        title = result.stdout.strip().split("\n")[0] if result.stdout.strip() else "Unknown"

        for ext in ["mp3", "m4a", "wav", "ogg", "webm"]:
            path = os.path.join(output_dir, f"yt_audio.{ext}")
            if os.path.exists(path):
                return path, title

        raise RuntimeError(f"Downloaded file not found in {output_dir}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("YouTube download timed out (5 min limit)")
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Audio → Transcript | MiMo V2.5 / Groq Whisper",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Provider selection ────────────────────────────────────────────────
        provider = st.radio(
            "Transcription Provider",
            options=["mimo", "groq"],
            format_func=lambda x: "🤖 MiMo V2.5 (OpenCode)" if x == "mimo" else "⚡ Groq Whisper",
            help="MiMo V2.5: Omnimodal ASR via OpenCode Go API. Groq: Fast Whisper-based transcription."
        )

        st.markdown("---")

        # ── API Keys (persisted via query params) ────────────────────────────
        # Read persisted keys from URL query params
        params = st.query_params

        if provider == "mimo":
            saved_mimo_key = params.get("mimo_key", "")
            api_key = st.text_input(
                "OpenCode API Key",
                value=saved_mimo_key,
                type="password",
                placeholder="sk-... (from opencode.ai/go)",
                help="Enter your OpenCode Go API key. Get one at opencode.ai/go"
            )
            # Persist key to URL params so it survives page reload
            if api_key:
                st.query_params["mimo_key"] = api_key
            elif "mimo_key" in st.query_params:
                del st.query_params["mimo_key"]

            groq_key = params.get("groq_key", "")

        else:  # groq
            saved_groq_key = params.get("groq_key", "")
            api_key = st.text_input(
                "Groq API Key",
                value=saved_groq_key,
                type="password",
                placeholder="gsk_... (from console.groq.com)",
                help="Enter your Groq API key. Get one at console.groq.com"
            )
            # Persist key to URL params so it survives page reload
            if api_key:
                st.query_params["groq_key"] = api_key
            elif "groq_key" in st.query_params:
                del st.query_params["groq_key"]

            mimo_key = params.get("mimo_key", "")

        st.markdown("---")

        language = st.selectbox(
            "Language",
            options=["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "hi", "ar", "pt", "ru"],
            index=0,
            help="Language hint for transcription. 'auto' lets the model detect."
        )

        st.markdown("---")
        st.markdown("""
        <small>
        <b>MiMo V2.5</b>: Omnimodal ASR (OpenCode Go API)<br/>
        <b>Groq Whisper</b>: Fast Whisper-large-v3 transcription<br/>
        Handles audio up to 1 hour with intelligent chunking<br/>
        Supports: MP3, WAV, M4A, OGG, WEBM, FLAC
        </small>
        """, unsafe_allow_html=True)

    # ── Main Area ────────────────────────────────────────────────────────────
    st.title("🎙️ Audio → Transcript")
    provider_label = "MiMo V2.5" if provider == "mimo" else "Groq Whisper"
    st.caption(f"Convert audio to text — powered by {provider_label}")

    # Warn if no API key
    if not api_key:
        key_name = "OpenCode" if provider == "mimo" else "Groq"
        st.warning(f"🔑 Please enter your {key_name} API key in the sidebar to continue.")
        st.stop()

    # ── Two-column layout (matching PDF-to-MD pattern) ──────────────────────
    col_input, col_output = st.columns([1, 1.3])

    with col_input:
        st.subheader("🎤 Audio Input")

        tab1, tab2, tab3 = st.tabs(["📁 Upload", "🎥 YouTube", "🎙️ Record"])

        wav_path = None
        source_label = ""
        raw_audio_path = None
        recorded_wav_bytes = None

        with tab1:
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["audio/*"],
                key="audio_upload",
                label_visibility="collapsed",
                help="Supports MP3, WAV, M4A, OGG, WEBM, FLAC, AAC (up to 200MB)"
            )
            if uploaded_file:
                raw_audio_path = save_uploaded_file(uploaded_file)
                source_label = f"Upload: {uploaded_file.name}"
                ext = Path(uploaded_file.name).suffix.lstrip(".")
                st.audio(uploaded_file, format=f"audio/{ext}")

        with tab2:
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://youtu.be/... or https://www.youtube.com/watch?v=...",
                key="yt_url"
            )

            st.caption("YouTube may require cookies for authentication.")
            cookies_file = st.file_uploader(
                "Upload cookies.txt (Netscape format)",
                type=["txt"],
                key="yt_cookies",
                help="Export cookies from your browser using a cookie extension in Netscape format"
            )

            cookies_path = None
            if cookies_file:
                cookies_path = os.path.join(tempfile.mkdtemp(), "cookies.txt")
                with open(cookies_path, "wb") as f:
                    f.write(cookies_file.getvalue())
                st.success("✅ Cookies loaded")

            if st.button("⬇️ Download Audio", key="yt_download", disabled=not yt_url):
                if yt_url:
                    with st.spinner("Downloading audio from YouTube... (this may take 1-2 minutes)"):
                        try:
                            audio_dl_path, video_title = download_youtube_audio(yt_url, cookies_path)
                            raw_audio_path = audio_dl_path
                            source_label = f"YouTube: {video_title}"
                            st.success(f"✅ Downloaded: {video_title}")
                            st.audio(audio_dl_path)
                        except RuntimeError as e:
                            st.error(f"❌ Download failed: {e}")

        with tab3:
            st.caption("Click the mic to start recording. Click again to stop.")
            try:
                from audio_recorder_streamlit import audio_recorder
                audio_bytes = audio_recorder(
                    sample_rate=16000,
                    energy_threshold=(-1.0, 1.0),
                    pause_threshold=600.0,
                )
                if audio_bytes:
                    # Save raw recording bytes to a temp WAV file
                    rec_dir = tempfile.mkdtemp(prefix="recording_")
                    rec_path = os.path.join(rec_dir, "recording.wav")
                    with open(rec_path, "wb") as f:
                        f.write(audio_bytes)

                    raw_audio_path = rec_path
                    source_label = "Recording (WAV)"
                    st.audio(audio_bytes, format="audio/wav")

                    # Provide .wav download
                    st.download_button(
                        "💾 Download recording as .wav",
                        data=audio_bytes,
                        file_name="recording.wav",
                        mime="audio/wav",
                        key="download_recording_wav"
                    )
            except ImportError:
                st.warning("audio-recorder-streamlit not installed. Using browser recorder fallback.")
                audio_value = st.audio_input(
                    "Click to record",
                    key="audio_record",
                    help="Click the microphone button to start recording"
                )
                if audio_value:
                    raw_audio_path = save_uploaded_file(audio_value)
                    actual_format = detect_audio_format(raw_audio_path)
                    source_label = f"Recording ({actual_format.upper()})"
                    st.audio(audio_value)

                    try:
                        wav_dl_path = convert_to_wav(raw_audio_path)
                        verify_fmt = detect_audio_format(wav_dl_path)
                        if verify_fmt == "wav":
                            with open(wav_dl_path, "rb") as f:
                                recorded_wav_bytes = f.read()
                    except Exception as e:
                        log.error(f"WAV conversion failed: {e}")

                    if recorded_wav_bytes:
                        st.download_button(
                            "💾 Download recording as .wav",
                            data=recorded_wav_bytes,
                            file_name="recording.wav",
                            mime="audio/wav",
                            key="download_recording_wav"
                        )

    # ── Convert & Show Info ──────────────────────────────────────────────────
    duration_min = 0.0
    if raw_audio_path:
        try:
            with st.spinner("Converting audio..."):
                wav_path = convert_to_wav(raw_audio_path)

            duration_min = get_audio_duration(wav_path)
            file_size_mb = os.path.getsize(raw_audio_path) / (1024 * 1024)

            st.success(f"**{source_label}** — {duration_min:.1f} min · {file_size_mb:.1f} MB")

            if duration_min > MAX_AUDIO_DURATION:
                st.warning(f"⚠️ Audio is {duration_min:.0f} minutes. Max supported: {MAX_AUDIO_DURATION} min. "
                          "Transcription may be incomplete.")
        except Exception as e:
            st.error(f"❌ Could not process audio: {e}")
            wav_path = None

    # ── Transcribe Button ────────────────────────────────────────────────────
    can_transcribe = wav_path is not None and bool(api_key)

    if st.button("🚀 Transcribe", type="primary", disabled=not can_transcribe, use_container_width=True):
        with col_output:
            st.subheader("📝 Transcript")

        progress_bar = st.progress(0.0, text="Preparing audio...")
        status_text = st.empty()

        def progress_callback(progress: float, message: str):
            progress_bar.progress(progress, text=message)
            status_text.info(message)

        start_time = time.time()

        try:
            transcript = transcribe_audio(
                wav_path,
                provider=provider,
                api_key=api_key,
                language=language,
                progress_callback=progress_callback
            )

            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text="✅ Transcription complete!")
            st.session_state.transcript = transcript

            method = f"{provider_label} — {duration_min:.1f} min audio — {elapsed:.1f}s"
            st.session_state.transcript_method = method

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Transcription failed: {e}")
            log.exception("Transcription error")

    # ── Output Display (matching PDF-to-MD pattern) ──────────────────────────
    with col_output:
        if st.session_state.transcript:
            st.subheader("📝 Transcript")

            transcript_text = st.session_state.transcript
            char_count = len(transcript_text)
            method = st.session_state.get("transcript_method", provider_label)
            st.caption(f"🔧 {method} · {char_count:,} chars")

            # Action buttons FIRST — always visible, no scrolling needed
            dl_cols = st.columns(2)
            with dl_cols[0]:
                st.download_button(
                    "📥 Download .txt",
                    data=transcript_text.encode("utf-8"),
                    file_name=TRANSCRIPT_FILENAME,
                    mime="text/plain",
                    use_container_width=True,
                    key="download_txt"
                )
            with dl_cols[1]:
                copy_button(transcript_text, "📋 Copy Transcript", key_suffix="transcript")

            # Second row: JSON download + editable toggle
            dl_cols2 = st.columns(2)
            with dl_cols2[0]:
                st.download_button(
                    "📄 Download .json",
                    data=json.dumps({
                        "source": source_label,
                        "duration_minutes": round(duration_min, 2),
                        "language": language,
                        "provider": provider,
                        "model": MIMO_MODEL if provider == "mimo" else GROQ_WHISPER_MODEL,
                        "transcript": transcript_text,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, indent=2, ensure_ascii=False),
                    file_name="transcript.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_json"
                )
            with dl_cols2[1]:
                # View toggle
                view = st.radio("Preview", ["Rendered", "Raw Text"], horizontal=True, key="view_toggle", label_visibility="collapsed")

            # Scrollable preview — fixed height, own scrollbar
            safe_transcript_display(transcript_text, view_mode=view)

        else:
            st.subheader("📝 Transcript")
            st.info("Transcript will appear here after you upload audio and click Transcribe.")

            with st.expander("💡 How it works"):
                st.markdown(
                    """
                    1. **Input**: Upload audio, paste a YouTube URL, or record from your mic
                    2. **Convert**: Audio is converted to 16 kHz mono WAV (ffmpeg or pure-Python fallback)
                    3. **Chunk**: Long audio is split into overlapping segments for reliable transcription
                    4. **Transcribe**: Each chunk is sent to your selected provider (MiMo V2.5 or Groq Whisper)
                    5. **Merge**: Overlapping chunks are stitched together with deduplication

                    **MiMo V2.5**: Omnimodal ASR model, handles long context well.
                    **Groq Whisper**: Ultra-fast transcription using Whisper-large-v3 on Groq's LPU infrastructure.

                    **API keys persist in your URL** — reload the page and they'll still be there.
                    """
                )


if __name__ == "__main__":
    main()
