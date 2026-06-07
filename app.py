#!/usr/bin/env python3
"""
Audio → Transcript — MiMo V2.5 ASR
====================================
Converts audio (uploaded, recorded, or from YouTube) to transcript.txt
using MiMo V2.5 omnimodal ASR via OpenCode Go API.

No pydub — uses ffmpeg directly (Python 3.13+ safe).
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
from pathlib import Path

import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

# ─── Configuration ────────────────────────────────────────────────────────────

OPCODE_BASE_URL = "https://opencode.ai/zen/go/v1"

# Audio chunking config
CHUNK_DURATION_MIN = int(os.environ.get("CHUNK_DURATION_MIN", "3"))
CHUNK_OVERLAP_SEC = int(os.environ.get("CHUNK_OVERLAP_SEC", "5"))
MAX_AUDIO_DURATION = 60  # minutes (1 hour)

# Output
TRANSCRIPT_FILENAME = "transcript.txt"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── OpenCode Go Client ──────────────────────────────────────────────────────

def get_opencode_client(api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at OpenCode Go API."""
    return OpenAI(
        base_url=OPCODE_BASE_URL,
        api_key=api_key,
    )


# ─── Audio Utilities (pydub-free, Python 3.13+ safe) ────────────────────────
# Uses ffmpeg when available, falls back to pure-Python WAV processing


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
    # Try ffprobe first
    if _check_ffmpeg():
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return float(result.stdout.strip()) / 60.0
            except ValueError:
                log.warning(f"ffprobe returned non-numeric duration: '{result.stdout.strip()}', falling back to WAV header")

    # Fallback: try reading WAV header directly
    try:
        header = _read_wav_header(filepath)
        if header["duration_sec"] > 0:
            return header["duration_sec"] / 60.0
    except (ValueError, struct.error) as e:
        log.warning(f"Could not read WAV header for duration: {e}")

    # Last resort: estimate from file size (assume 16kHz mono 16-bit = 32000 bytes/sec)
    try:
        file_size = os.path.getsize(filepath)
        # For 16kHz mono 16-bit WAV: 32000 bytes per second of audio
        # Subtract 44 bytes for WAV header
        estimated_sec = max(0, (file_size - 44) / 32000)
        if estimated_sec > 0:
            log.info(f"Estimated duration from file size: {estimated_sec:.1f}s")
            return estimated_sec / 60.0
    except OSError:
        pass

    return 0.0


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
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 44:
            # Verify the output is actually a valid WAV
            try:
                fmt = detect_audio_format(output_path)
                if fmt == "wav":
                    return output_path
                else:
                    log.warning(f"ffmpeg output is {fmt}, not WAV — trying fallback")
            except Exception:
                pass
        elif result.returncode != 0:
            log.warning(f"ffmpeg conversion failed, trying WAV fallback: {result.stderr[-200:]}")
        else:
            log.warning("ffmpeg produced empty output, trying WAV fallback")

    # Fallback: try pure-Python WAV processing
    try:
        header = _read_wav_header(input_path)
        if header["frame_rate"] == 16000 and header["channels"] == 1 and header["bits_per_sample"] == 16:
            shutil.copy2(input_path, output_path)
            return output_path
        return _resample_wav(input_path, output_path)
    except (ValueError, struct.error) as e:
        raise RuntimeError(
            f"Cannot convert {input_path}: ffmpeg conversion failed and the file is not a valid WAV. "
            f"Error: {e}"
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
    st.audio_input often names files .wav even though content is WebM/Opus.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="upload_")

    # Save with original suffix first
    original_suffix = Path(uploaded_file.name).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=original_suffix, dir=output_dir)
    os.close(fd)

    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Detect actual format and rename if the extension is wrong
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
    elif header[:4] == b"\x1a\x45\xdf\xa5":  # WebM/Matroska
        return "webm"
    elif header[4:8] == b"ftyp":  # MP4/M4A
        return "m4a"
    else:
        return "unknown"


# ─── Transcription Engine ────────────────────────────────────────────────────

def transcribe_chunk(wav_path: str, api_key: str, model: str,
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
                model=model,
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
            log.error(f"Transcription API error (attempt {attempt+1}): {e}")
            if "image input" in error_str.lower() or "no endpoints found" in error_str.lower():
                return (
                    "[TRANSCRIPTION_ERROR: The selected model does not support audio input. "
                    "Please switch to 'mimo-v2-omni' in the sidebar Settings.]"
                )
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return f"[TRANSCRIPTION_ERROR: {e}]"


def transcribe_audio(wav_path: str, api_key: str, model: str,
                     language: str = "en", progress_callback=None) -> str:
    """Transcribe a full audio file (handles chunking for long audio)."""
    duration_min = get_audio_duration(wav_path)
    log.info(f"Transcribing audio: {duration_min:.1f} minutes")

    if duration_min <= 5:
        if progress_callback:
            progress_callback(0.1, "Transcribing audio...")
        transcript = transcribe_chunk(wav_path, api_key=api_key, model=model, language=language)
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
        part = transcribe_chunk(chunk_path, api_key=api_key, model=model,
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
        page_title="Audio → Transcript | MiMo V2.5 ASR",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("### 🔑 API Key")
        api_key = st.text_input(
            "OpenCode API Key",
            value="",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenCode Go API key. Get one at opencode.ai"
        )

        model_choice = st.selectbox(
            "Model",
            options=["mimo-v2-omni", "mimo-v2.5"],
            index=0,
            help="mimo-v2-omni: Best for audio (recommended). mimo-v2.5: Also supports audio."
        )

        language = st.selectbox(
            "Language",
            options=["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "hi", "ar", "pt", "ru"],
            index=0,
            help="Language hint for transcription. 'auto' lets the model detect."
        )

        st.markdown("---")
        st.markdown("""
        <small>
        Powered by <b>MiMo V2.5</b> via OpenCode Go API<br/>
        Handles audio up to 1 hour with intelligent chunking<br/>
        Supports: MP3, WAV, M4A, OGG, WEBM, FLAC
        </small>
        """, unsafe_allow_html=True)

    # ── Main Area ────────────────────────────────────────────────────────────
    st.title("🎙️ Audio → Transcript")
    st.caption("Convert audio to text — powered by MiMo V2.5 ASR")

    # Warn if no API key
    if not api_key:
        st.warning("🔑 Please enter your OpenCode API key in the sidebar to continue.")
        st.stop()

    # ── Step 1: Audio Input ──────────────────────────────────────────────────
    st.markdown("## 🎤 Audio Input")

    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎥 YouTube URL", "🎙️ Record Audio"])

    wav_path = None
    source_label = ""
    raw_audio_path = None

    with tab1:
        st.markdown("### Upload an audio file")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "webm", "flac", "aac"],
            key="audio_upload",
            help="Supports MP3, WAV, M4A, OGG, WEBM, FLAC, AAC (up to 200MB)"
        )
        if uploaded_file:
            raw_audio_path = save_uploaded_file(uploaded_file)
            source_label = f"Upload: {uploaded_file.name}"
            ext = Path(uploaded_file.name).suffix.lstrip(".")
            st.audio(uploaded_file, format=f"audio/{ext}")

    with tab2:
        st.markdown("### Download audio from YouTube")
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

        yt_col1, yt_col2 = st.columns([1, 3])
        with yt_col1:
            download_btn = st.button("⬇️ Download Audio", key="yt_download",
                                     disabled=not yt_url)
        with yt_col2:
            if download_btn and yt_url:
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
        st.markdown("### Record audio from your microphone")
        st.caption(
            "**How to use:** Click the mic icon to START recording (it turns red). "
            "Speak, then click again to STOP. The recording will appear below."
        )
        # energy_threshold=(-1.0, 1.0) disables voice-activity auto-stop
        # so recording only stops when user clicks again (avoids first-click bug)
        # sample_rate=16000 records at 16kHz directly (matches ASR requirements)
        wav_bytes = audio_recorder(
            text="",
            energy_threshold=(-1.0, 1.0),
            pause_threshold=600.0,  # 10 min max — won't auto-stop
            icon_name="microphone",
            icon_size="2x",
            sample_rate=16000,
            key="audio_recorder"
        )
        if wav_bytes and len(wav_bytes) > 44:  # More than just a WAV header
            source_label = "Recording (WAV 16kHz)"

            # Save raw WAV bytes to a temp file
            fd, recording_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with open(recording_path, "wb") as f:
                f.write(wav_bytes)
            raw_audio_path = recording_path

            # Play back the recording
            st.audio(wav_bytes, format="audio/wav")

            # Convert to mono 16kHz WAV for a clean download
            # (audio_recorder produces stereo; convert_to_wav makes mono)
            try:
                converted_wav_path = convert_to_wav(recording_path)
                with open(converted_wav_path, "rb") as f:
                    download_wav_bytes = f.read()
                st.download_button(
                    "💾 Download recording as .wav (16kHz mono)",
                    data=download_wav_bytes,
                    file_name="recording.wav",
                    mime="audio/wav",
                    key="download_recording_wav"
                )
            except Exception as e:
                # Fallback: offer original bytes
                log.warning(f"WAV conversion for download failed: {e}")
                st.download_button(
                    "💾 Download recording as .wav (original)",
                    data=wav_bytes,
                    file_name="recording.wav",
                    mime="audio/wav",
                    key="download_recording_wav"
                )
        elif wav_bytes:
            st.warning("⚠️ Recording too short — please try again and speak longer before clicking stop.")

    # ── Convert & Show Info ──────────────────────────────────────────────────
    st.markdown("---")

    duration_min = 0.0
    if raw_audio_path:
        try:
            with st.spinner("Converting audio..."):
                wav_path = convert_to_wav(raw_audio_path)

            duration_min = get_audio_duration(wav_path)
            file_size_mb = os.path.getsize(raw_audio_path) / (1024 * 1024)

            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{duration_min:.1f} min")
            col2.metric("File Size", f"{file_size_mb:.1f} MB")
            col3.metric("Est. Chunks", f"{max(1, int(duration_min / 4) + 1)}")

            if duration_min > MAX_AUDIO_DURATION:
                st.warning(f"⚠️ Audio is {duration_min:.0f} minutes. Max supported: {MAX_AUDIO_DURATION} min. "
                          "Transcription may be incomplete.")
        except Exception as e:
            st.error(f"❌ Could not process audio: {e}")
            wav_path = None

    # ── Step 2: Transcribe ──────────────────────────────────────────────────
    if wav_path:
        st.markdown("## 🚀 Transcribe")

        if st.button("🚀 Start Transcription", type="primary", key="transcribe_btn"):
            progress_bar = st.progress(0.0, text="Preparing audio...")
            status_text = st.empty()

            def progress_callback(progress: float, message: str):
                progress_bar.progress(progress, text=message)
                status_text.info(message)

            start_time = time.time()

            try:
                transcript = transcribe_audio(
                    wav_path,
                    api_key=api_key,
                    model=model_choice,
                    language=language,
                    progress_callback=progress_callback
                )

                elapsed = time.time() - start_time
                progress_bar.progress(1.0, text="✅ Transcription complete!")
                st.session_state.transcript = transcript

                st.info(f"Source: {source_label} | Duration: {duration_min:.1f} min | Time: {elapsed:.1f}s")

            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Transcription failed: {e}")
                log.exception("Transcription error")

    # ── Show Transcript ─────────────────────────────────────────────────────
    if st.session_state.transcript:
        st.markdown("## 📝 Transcript")
        edited_transcript = st.text_area(
            "Transcript (editable)",
            value=st.session_state.transcript,
            height=400,
            key="transcript_output"
        )
        st.session_state.transcript = edited_transcript

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download transcript.txt",
                data=edited_transcript.encode("utf-8"),
                file_name=TRANSCRIPT_FILENAME,
                mime="text/plain",
                key="download_txt"
            )
        with col2:
            st.download_button(
                "📋 Download transcript.json",
                data=json.dumps({
                    "source": source_label,
                    "duration_minutes": round(duration_min, 2),
                    "language": language,
                    "model": model_choice,
                    "transcript": edited_transcript,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, indent=2, ensure_ascii=False),
                file_name="transcript.json",
                mime="application/json",
                key="download_json"
            )

    elif not raw_audio_path:
        st.info("👆 Upload an audio file, enter a YouTube URL, or record audio to get started.")


if __name__ == "__main__":
    main()
