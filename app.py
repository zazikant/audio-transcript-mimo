#!/usr/bin/env python3
"""
Audio-to-Transcript Streamlit App
==================================
Converts audio (uploaded, recorded, or from YouTube) to transcript.txt
using MiMo V2.5 multimodal ASR via OpenCode Go API.

Handles long audio up to 1 hour via intelligent chunking.
"""

import os
import io
import re
import json
import time
import base64
import tempfile
import logging
from pathlib import Path

import streamlit as st
import httpx
from pydub import AudioSegment
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

OPCODE_API_KEY = os.environ.get(
    "OPENCODE_API_KEY",
    "sk-ljxfsgwP7tLRFfTIgmMOd7n4vBcXNTM34UAVR30mMCNMU28F9MsHJTKKssdqdhnR"
)
OPCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5")

# Audio chunking config
CHUNK_DURATION_MIN = int(os.environ.get("CHUNK_DURATION_MIN", "3"))   # minutes
CHUNK_OVERLAP_SEC  = int(os.environ.get("CHUNK_OVERLAP_SEC", "5"))   # seconds
MAX_AUDIO_DURATION  = 60   # minutes (1 hour)
MAX_FILE_SIZE_MB    = 25   # OpenAI-compatible API limit per request

# Output
TRANSCRIPT_FILENAME = "transcript.txt"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── OpenCode Go Client ──────────────────────────────────────────────────────

@st.cache_resource
def get_opencode_client() -> OpenAI:
    """Create an OpenAI client pointed at OpenCode Go API."""
    return OpenAI(
        base_url=OPCODE_BASE_URL,
        api_key=OPCODE_API_KEY,
    )


# ─── Audio Utilities ─────────────────────────────────────────────────────────

def convert_to_wav(audio_bytes: bytes, source_format: str = "auto") -> AudioSegment:
    """Convert any audio format to a standardized WAV AudioSegment.

    Args:
        audio_bytes: Raw audio bytes.
        source_format: pydub-compatible format string (mp3, wav, m4a, ogg, webm, etc.)

    Returns:
        AudioSegment in 16kHz mono WAV.
    """
    if source_format == "auto":
        # Try common formats
        for fmt in ["wav", "mp3", "m4a", "ogg", "webm", "flac"]:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                break
            except Exception:
                continue
        else:
            # Last resort: let pydub auto-detect
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    else:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)

    # Standardize: 16kHz, mono, 16-bit PCM
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio


def get_audio_duration(audio: AudioSegment) -> float:
    """Return duration in minutes."""
    return len(audio) / 1000.0 / 60.0


def chunk_audio(audio: AudioSegment, chunk_minutes: int = CHUNK_DURATION_MIN,
                overlap_seconds: int = CHUNK_OVERLAP_SEC) -> list[AudioSegment]:
    """Split audio into overlapping chunks for long-form transcription.

    Args:
        audio: The full AudioSegment.
        chunk_minutes: Duration of each chunk in minutes.
        overlap_seconds: Overlap between chunks to avoid cutting words.

    Returns:
        List of AudioSegment chunks.
    """
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = overlap_seconds * 1000
    total_ms = len(audio)

    if total_ms <= chunk_ms:
        return [audio]

    chunks = []
    start = 0
    while start < total_ms:
        end = min(start + chunk_ms, total_ms)
        chunks.append(audio[start:end])
        if end >= total_ms:
            break
        start = end - overlap_ms  # overlap to catch cut words

    return chunks


def audio_segment_to_base64(audio: AudioSegment) -> str:
    """Export AudioSegment as WAV and return base64-encoded string."""
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def estimate_chunk_size_b64(audio: AudioSegment) -> int:
    """Estimate the base64 size of an audio chunk in bytes."""
    # WAV: 16000 samples/sec * 2 bytes/sample = 32KB/sec
    # Base64 adds ~33% overhead
    wav_size = len(audio) / 1000.0 * 32000
    return int(wav_size * 1.34)


# ─── Transcription Engine ────────────────────────────────────────────────────

def transcribe_chunk(audio: AudioSegment, language: str = "en",
                     prompt_context: str = "") -> str:
    """Transcribe a single audio chunk using MiMo V2.5 via OpenCode Go API.

    Args:
        audio: AudioSegment chunk (max ~5 minutes recommended).
        language: Language hint (en, zh, auto, etc.).
        prompt_context: Previous transcript text for continuity.

    Returns:
        Transcribed text.
    """
    client = get_opencode_client()

    audio_b64 = audio_segment_to_base64(audio)
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    # Build system message
    system_msg = (
        "You are a professional speech-to-text transcription engine. "
        "Transcribe the provided audio accurately and faithfully. "
        "Output ONLY the transcript text — no timestamps, no commentary, no labels. "
        "Preserve speaker turns with newlines. "
        "If the audio is not speech (music, noise, silence), respond with: [non-speech]"
    )

    # Build user message
    user_parts = [
        {"type": "text", "text": f"Transcribe this audio. Language: {language}."},
    ]
    if prompt_context:
        user_parts[0]["text"] += (
            f"\n\nContext from previous chunk (for continuity):\n{prompt_context[-500:]}"
        )

    user_parts.append({
        "type": "input_audio",
        "input_audio": {"data": audio_data_uri, "format": "wav"}
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

            # Try content first
            if msg.content:
                result = msg.content.strip()

            # Fallback: reasoning_content (some models put output here)
            if not result:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    result = rc.strip()

            if not result:
                if attempt < max_retries - 1:
                    log.warning(f"Empty API response, retry {attempt+1}/{max_retries}")
                    time.sleep(2 * (attempt + 1))
                    continue
                return "[TRANSCRIPTION_ERROR: Empty response from API]"

            return result

        except Exception as e:
            log.error(f"Transcription API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return f"[TRANSCRIPTION_ERROR: {e}]"


def transcribe_audio(audio: AudioSegment, language: str = "en",
                     progress_callback=None) -> str:
    """Transcribe a full audio (handles chunking for long audio).

    Args:
        audio: Full AudioSegment.
        language: Language hint.
        progress_callback: Optional callable(progress: float, message: str).

    Returns:
        Full transcript text.
    """
    duration_min = get_audio_duration(audio)
    log.info(f"Transcribing audio: {duration_min:.1f} minutes")

    # Determine chunking strategy based on duration
    if duration_min <= 5:
        # Short audio: single chunk
        if progress_callback:
            progress_callback(0.1, "Transcribing audio...")
        transcript = transcribe_chunk(audio, language=language)
        if progress_callback:
            progress_callback(1.0, "Done!")
        return transcript

    # Long audio: chunk with overlap
    # Adjust chunk size based on duration to stay within API limits
    if duration_min <= 15:
        chunk_min = 5
    elif duration_min <= 30:
        chunk_min = 4
    else:
        chunk_min = 3

    chunks = chunk_audio(audio, chunk_minutes=chunk_min)
    log.info(f"Split into {len(chunks)} chunks of ~{chunk_min} min")

    transcript_parts = []
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        msg = f"Transcribing chunk {i+1}/{len(chunks)}..."
        if progress_callback:
            progress_callback(progress * 0.9, msg)

        # Pass previous context for continuity
        context = " ".join(transcript_parts[-2:]) if transcript_parts else ""
        part = transcribe_chunk(chunk, language=language, prompt_context=context)

        if part and part != "[non-speech]":
            transcript_parts.append(part)

        # Small delay between API calls
        if i < len(chunks) - 1:
            time.sleep(1)

    # Merge overlapping text (simple dedup of overlap region)
    full_transcript = merge_transcript_parts(transcript_parts)

    if progress_callback:
        progress_callback(1.0, "Transcription complete!")

    return full_transcript


def merge_transcript_parts(parts: list[str]) -> str:
    """Merge transcript chunks, handling overlap deduplication.

    Since we use overlapping chunks, the end of one chunk may overlap
    with the beginning of the next. We try to find and remove duplicate text.
    """
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    merged = parts[0]
    for i in range(1, len(parts)):
        next_part = parts[i]
        # Try to find overlap: check if end of merged matches start of next
        overlap_found = False
        # Check last 50 words of merged against first 50 words of next
        merged_words = merged.split()
        next_words = next_part.split()

        check_len = min(30, len(merged_words), len(next_words))
        if check_len > 3:
            for window in range(check_len, 2, -1):
                merged_tail = " ".join(merged_words[-window:]).lower().strip(".,!?;:")
                next_head = " ".join(next_words[:window]).lower().strip(".,!?;:")
                if merged_tail == next_head and len(merged_tail) > 10:
                    # Found overlap — skip the duplicate part
                    merged = merged + " " + " ".join(next_words[window:])
                    overlap_found = True
                    break

        if not overlap_found:
            merged = merged + "\n\n" + next_part

    return merged.strip()


# ─── YouTube Download ────────────────────────────────────────────────────────

def download_youtube_audio(url: str, output_dir: str = None) -> tuple[str, str]:
    """Download audio from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the audio file.

    Returns:
        Tuple of (file_path, video_title).

    Raises:
        RuntimeError: If download fails.
    """
    import subprocess

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--no-playlist",
        "--print", "title",
        url
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[-500:]}")

        title = result.stdout.strip().split("\n")[0] if result.stdout.strip() else "Unknown"

        # Find the downloaded file
        for ext in ["mp3", "m4a", "wav", "ogg", "webm"]:
            path = os.path.join(output_dir, f"yt_audio.{ext}")
            if os.path.exists(path):
                return path, title

        raise RuntimeError(f"Downloaded file not found in {output_dir}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("YouTube download timed out (5 min limit)")
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp not installed. Install with: pip install yt-dlp\n"
            "Also ensure ffmpeg is available: apt install ffmpeg"
        )


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Audio → Transcript | MiMo V2.5 ASR",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        language = st.selectbox(
            "Language",
            options=["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "hi", "ar", "pt", "ru"],
            index=0,
            help="Language hint for transcription. 'auto' lets the model detect."
        )

        st.markdown("---")
        st.markdown("### 📋 Pipeline")
        st.markdown("""
        1. **Input**: Upload / Record / YouTube URL
        2. **Process**: Convert → Chunk → Transcribe
        3. **Output**: `transcript.txt` download
        """)

        st.markdown("---")
        st.markdown("### 🔧 API Config")
        api_key = st.text_input(
            "OpenCode API Key",
            value=OPCODE_API_KEY,
            type="password",
            help="Your OpenCode Go API key"
        )
        if api_key != OPCODE_API_KEY:
            os.environ["OPENCODE_API_KEY"] = api_key
            # Clear cached client
            if "get_opencode_client" in st.session_state:
                del st.session_state["get_opencode_client"]

        model_choice = st.selectbox(
            "Model",
            options=["mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-omni"],
            index=0,
            help="MiMo model to use for transcription"
        )
        os.environ["MIMO_MODEL"] = model_choice

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
    st.caption("Transcribe audio files, recordings, or YouTube videos using MiMo V2.5 ASR")

    # Input mode tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎥 YouTube URL", "🎙️ Record Audio"])

    audio_bytes = None
    source_format = "auto"
    source_label = ""

    with tab1:
        st.markdown("### Upload an audio file")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "webm", "flac", "aac"],
            key="audio_upload",
            help="Supports MP3, WAV, M4A, OGG, WEBM, FLAC, AAC (up to 200MB)"
        )
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            ext = Path(uploaded_file.name).suffix.lstrip(".")
            source_format = ext if ext else "auto"
            source_label = f"Upload: {uploaded_file.name}"
            st.audio(uploaded_file, format=f"audio/{ext}")

    with tab2:
        st.markdown("### Download audio from YouTube")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtu.be/... or https://www.youtube.com/watch?v=...",
            key="yt_url"
        )
        yt_col1, yt_col2 = st.columns([1, 3])
        with yt_col1:
            download_btn = st.button("⬇️ Download Audio", key="yt_download",
                                     disabled=not yt_url)
        with yt_col2:
            if download_btn and yt_url:
                with st.spinner("Downloading audio from YouTube... (this may take 1-2 minutes)"):
                    try:
                        audio_path, video_title = download_youtube_audio(yt_url)
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        source_format = Path(audio_path).suffix.lstrip(".")
                        source_label = f"YouTube: {video_title}"
                        st.success(f"✅ Downloaded: {video_title}")
                        st.audio(audio_path)
                    except RuntimeError as e:
                        st.error(f"❌ Download failed: {e}")

    with tab3:
        st.markdown("### Record audio from your microphone")
        audio_value = st.audio_input(
            "Click to record",
            key="audio_record",
            help="Click the microphone button to start recording"
        )
        if audio_value:
            audio_bytes = audio_value.read()
            source_format = "wav"  # Streamlit audio_input returns WAV
            source_label = "Recording"
            st.audio(audio_value)

    # ── Transcription Section ────────────────────────────────────────────────
    st.markdown("---")

    if audio_bytes:
        # Show audio info
        try:
            audio = convert_to_wav(audio_bytes, source_format)
            duration_min = get_audio_duration(audio)
            file_size_mb = len(audio_bytes) / (1024 * 1024)

            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{duration_min:.1f} min")
            col2.metric("File Size", f"{file_size_mb:.1f} MB")
            col3.metric("Est. Chunks", f"{max(1, int(duration_min / 4) + 1)}")

            if duration_min > MAX_AUDIO_DURATION:
                st.warning(f"⚠️ Audio is {duration_min:.0f} minutes. Max supported: {MAX_AUDIO_DURATION} min. "
                          "Transcription may be incomplete.")

            # Transcribe button
            if st.button("🚀 Transcribe", type="primary", key="transcribe_btn"):
                progress_bar = st.progress(0.0, text="Preparing audio...")
                status_text = st.empty()

                def progress_callback(progress: float, message: str):
                    progress_bar.progress(progress, text=message)
                    status_text.info(message)

                start_time = time.time()

                try:
                    transcript = transcribe_audio(
                        audio,
                        language=language,
                        progress_callback=progress_callback
                    )

                    elapsed = time.time() - start_time
                    progress_bar.progress(1.0, text="✅ Transcription complete!")

                    # Show results
                    st.markdown("---")
                    st.markdown(f"### 📝 Transcript ({elapsed:.1f}s)")
                    st.info(f"Source: {source_label} | Duration: {duration_min:.1f} min | Time: {elapsed:.1f}s")

                    # Editable transcript
                    edited_transcript = st.text_area(
                        "Transcript (editable)",
                        value=transcript,
                        height=400,
                        key="transcript_output"
                    )

                    # Download buttons
                    col1, col2, col3 = st.columns(3)

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

                    with col3:
                        # Save to RAG Document Assistant
                        if st.button("🧠 Save to Second Brain", key="save_rag"):
                            try:
                                rag_resp = httpx.post(
                                    "https://rag-document-assistant-three.vercel.app/api/upload",
                                    json={
                                        "type": "text",
                                        "content": edited_transcript,
                                        "name": f"transcript_{int(time.time())}.txt",
                                        "mode": "Add"
                                    },
                                    timeout=30
                                )
                                if rag_resp.status_code == 200:
                                    st.success("✅ Saved to Second Brain (RAG)!")
                                else:
                                    st.warning(f"RAG save returned: {rag_resp.status_code}")
                            except Exception as e:
                                st.warning(f"RAG save failed: {e}")

                    # Send to Circular Dependencies Agent for cleanup
                    with st.expander("🔄 Clean with Circular Dependencies Agent"):
                        if st.button("Run Circular Dependencies Cleanup", key="cd_cleanup"):
                            with st.spinner("Cleaning transcript structure..."):
                                try:
                                    cd_resp = httpx.post(
                                        "https://ax-opencode-translator.vercel.app/api/translate",
                                        json={
                                            "text": (
                                                "Convert telegraphic notes into a structured, "
                                                "circular dependicies removal. Preserve Facts, "
                                                "headings, subheadings, bullet points. Add "
                                                "Argumentative connectives and logical flow. "
                                                "Style polished.\n\nInput:\n\n"
                                                + edited_transcript
                                            ),
                                            "sourceLanguage": "en",
                                            "targetLanguage": "en",
                                            "fast": True
                                        },
                                        timeout=60
                                    )
                                    if cd_resp.status_code == 200:
                                        cleaned = cd_resp.json().get("translatedText", "")
                                        st.text_area("Cleaned Transcript", value=cleaned, height=300,
                                                    key="cd_output")
                                        st.download_button(
                                            "Download cleaned.txt",
                                            data=cleaned.encode("utf-8"),
                                            file_name="transcript_cleaned.txt",
                                            mime="text/plain",
                                            key="download_cleaned"
                                        )
                                    else:
                                        st.error(f"Cleanup failed: {cd_resp.status_code}")
                                except Exception as e:
                                    st.error(f"Cleanup error: {e}")

                    # Send to Atomic Graph Agent for knowledge graph
                    with st.expander("🧬 Generate Atomic Graph (Knowledge Map)"):
                        if st.button("Build Knowledge Graph", key="ag_build"):
                            with st.spinner("Building knowledge graph from transcript..."):
                                try:
                                    system_prompt = (
                                        "You are a semantic reasoning engine that builds "
                                        "knowledge graphs from raw thinking. You do NOT merely "
                                        "reformat or summarise — you REASON through the semantic "
                                        "space of ideas. CRITICAL OUTPUT RULES: Always respond "
                                        "with valid JSON only. No markdown, no explanation, no "
                                        "code fences."
                                    )

                                    # Step 1: EXTRACT
                                    extract_prompt = (
                                        "Extract atomic concepts from these notes. Each concept = "
                                        "ONE idea only. Return JSON: "
                                        '{ "nodes": [{ "id": "c1", "title": "...", '
                                        '"summary": "...", "tags": ["..."] }] }\n\n'
                                        f"Raw notes:\n{edited_transcript[:5000]}"
                                    )

                                    ag_resp = httpx.post(
                                        "https://atomic-graph.vercel.app/api/nvidia",
                                        json={
                                            "apiKey": "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",
                                            "model": "openai/gpt-oss-120b",
                                            "messages": [
                                                {"role": "system", "content": system_prompt},
                                                {"role": "user", "content": extract_prompt}
                                            ]
                                        },
                                        timeout=120
                                    )
                                    if ag_resp.status_code == 200:
                                        ag_data = ag_resp.json()
                                        content = ag_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                                        # Try to parse as JSON
                                        try:
                                            graph = json.loads(content)
                                            st.json(graph)
                                        except json.JSONDecodeError:
                                            st.text_area("Graph Output (raw)", value=content, height=300,
                                                        key="ag_output_raw")
                                    else:
                                        st.error(f"Atomic Graph failed: {ag_resp.status_code}")
                                except Exception as e:
                                    st.error(f"Atomic Graph error: {e}")

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Transcription failed: {e}")
                    log.exception("Transcription error")

        except Exception as e:
            st.error(f"❌ Could not process audio: {e}")
            log.exception("Audio processing error")

    else:
        st.info("👆 Upload an audio file, enter a YouTube URL, or record audio to get started.")


if __name__ == "__main__":
    main()
