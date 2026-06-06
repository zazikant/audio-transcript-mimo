#!/usr/bin/env python3
"""
Audio-to-Transcript Pipeline — MiMo V2.5 ASR
==============================================
Converts audio (uploaded, recorded, or from YouTube) to transcript.txt
using MiMo V2.5 omnimodal ASR via OpenCode Go API.

Full Pipeline:
  1. Audio Input → Transcription (MiMo V2.5)
  2. Seed Research → Deep Research (adversarial)
  3. Circular Dependencies Cleanup
  4. Atomic Graph (Knowledge Map)
  5. Second Brain RAG Storage

No pydub — uses ffmpeg directly (Python 3.13+ safe).
"""

import os
import io
import re
import json
import time
import base64
import tempfile
import logging
import subprocess
from pathlib import Path

import streamlit as st
import httpx
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

OPCODE_API_KEY = os.environ.get(
    "OPENCODE_API_KEY",
    "sk-ljxfsgwP7tLRFfTIgmMOd7n4vBcXNTM34UAVR30mMCNMU28F9MsHJTKKssdqdhnR"
)
OPCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5")

# Audio chunking config
CHUNK_DURATION_MIN = int(os.environ.get("CHUNK_DURATION_MIN", "3"))
CHUNK_OVERLAP_SEC = int(os.environ.get("CHUNK_OVERLAP_SEC", "5"))
MAX_AUDIO_DURATION = 60  # minutes (1 hour)

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


# ─── FFmpeg Audio Utilities (pydub-free, Python 3.13+ safe) ──────────────────

def get_audio_duration(filepath: str) -> float:
    """Return audio duration in minutes using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip()) / 60.0


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """Convert any audio file to 16kHz mono 16-bit PCM WAV using ffmpeg.

    Args:
        input_path: Path to source audio file.
        output_path: Path for output WAV. If None, creates a temp file.

    Returns:
        Path to the converted WAV file.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",       # 16kHz sample rate
        "-ac", "1",           # mono
        "-sample_fmt", "s16", # 16-bit PCM
        "-f", "wav",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")
    return output_path


def chunk_audio_wav(input_wav: str, chunk_minutes: int = CHUNK_DURATION_MIN,
                    overlap_seconds: int = CHUNK_OVERLAP_SEC,
                    output_dir: str = None) -> list[str]:
    """Split a WAV file into overlapping chunks using ffmpeg.

    Args:
        input_wav: Path to the standardized WAV file.
        chunk_minutes: Duration of each chunk in minutes.
        overlap_seconds: Overlap between chunks.
        output_dir: Directory for chunk files. If None, creates temp dir.

    Returns:
        List of file paths to chunk WAV files.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")

    total_duration_min = get_audio_duration(input_wav)
    chunk_sec = chunk_minutes * 60
    overlap_sec = overlap_seconds
    total_sec = total_duration_min * 60

    if total_sec <= chunk_sec:
        return [input_wav]

    chunks = []
    start = 0
    idx = 0
    while start < total_sec:
        end = min(start + chunk_sec, total_sec)
        out_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_wav,
            "-ss", str(start),
            "-to", str(end),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            "-f", "wav", out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            chunks.append(out_path)
        else:
            log.warning(f"Chunk {idx} extraction failed: {result.stderr[:200]}")

        if end >= total_sec:
            break
        start = end - overlap_sec
        idx += 1

    return chunks


def wav_to_base64(filepath: str) -> str:
    """Read a WAV file and return base64-encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_uploaded_file(uploaded_file, output_dir: str = None) -> str:
    """Save a Streamlit UploadedFile to disk and return its path."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="upload_")

    suffix = Path(uploaded_file.name).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=output_dir)
    os.close(fd)

    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return path


# ─── Transcription Engine ────────────────────────────────────────────────────

def transcribe_chunk(wav_path: str, language: str = "en",
                     prompt_context: str = "") -> str:
    """Transcribe a single audio chunk using MiMo V2.5 via OpenCode Go API.

    Uses the audio_url content format with base64 data URI.
    """
    client = get_opencode_client()

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

            # Try content first
            if msg.content:
                result = msg.content.strip()

            # Fallback: reasoning_content (MiMo puts output here sometimes)
            if not result:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    # Extract actual transcript from reasoning
                    # Reasoning often contains the transcript at the end
                    lines = rc.strip().split("\n")
                    # Find lines that look like actual transcript (not reasoning)
                    transcript_lines = []
                    skip_patterns = ["The user", "Let me", "I need to", "This is",
                                     "I'll", "Looking at", "Checking", "Analyzing",
                                     "The audio", "I can hear", "I should"]
                    for line in lines:
                        stripped = line.strip()
                        if stripped and not any(stripped.startswith(p) for p in skip_patterns):
                            # Check if it looks like a transcript quote
                            if stripped.startswith('"') or stripped.startswith("'"):
                                # Remove quotes
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
            log.error(f"Transcription API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return f"[TRANSCRIPTION_ERROR: {e}]"


def transcribe_audio(wav_path: str, language: str = "en",
                     progress_callback=None) -> str:
    """Transcribe a full audio file (handles chunking for long audio)."""
    duration_min = get_audio_duration(wav_path)
    log.info(f"Transcribing audio: {duration_min:.1f} minutes")

    if duration_min <= 5:
        if progress_callback:
            progress_callback(0.1, "Transcribing audio...")
        transcript = transcribe_chunk(wav_path, language=language)
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
        part = transcribe_chunk(chunk_path, language=language, prompt_context=context)

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
    """Download audio from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL.
        cookies_path: Optional path to cookies file for authentication.

    Returns:
        Tuple of (file_path, video_title).
    """
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
                    "file (Netscape format) using the 'Upload Cookies' button below. "
                    "See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
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
        raise RuntimeError(
            "yt-dlp not installed. Install with: pip install yt-dlp"
        )


# ─── Pipeline: Seed Research ─────────────────────────────────────────────────

def run_seed_research(transcript: str, api_key: str) -> dict:
    """Run seed research on the transcript to identify key topics and questions.

    Uses MiMo V2.5 to analyze the transcript and generate research seeds.
    """
    client = OpenAI(base_url=OPCODE_BASE_URL, api_key=api_key)

    prompt = f"""Analyze the following transcript and generate a research seed document.

For each major topic discussed, identify:
1. **Core Claim**: What is the main assertion?
2. **Evidence Provided**: What evidence supports it?
3. **Open Questions**: What questions remain unanswered?
4. **Contradictions**: Are there internal contradictions or tensions?
5. **Research Directions**: What further investigation would be valuable?

Also provide:
- A 2-3 sentence executive summary
- Top 5 key entities (people, organizations, concepts)
- Suggested search queries for deep research

TRANSCRIPT:
{transcript[:8000]}"""

    try:
        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "system", "content": "You are a research analyst. Generate structured research seeds from transcripts. Output in clean markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.3,
        )
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", "") or ""
        return {"status": "success", "content": content.strip()}
    except Exception as e:
        return {"status": "error", "content": f"Seed research failed: {e}"}


# ─── Pipeline: Deep Research (Adversarial) ───────────────────────────────────

def run_deep_research(seed_doc: str, api_key: str) -> dict:
    """Run adversarial deep research based on seed research findings.

    Uses MiMo V2.5 with adversarial prompting to stress-test claims.
    """
    client = OpenAI(base_url=OPCODE_BASE_URL, api_key=api_key)

    prompt = f"""You are an adversarial research critic. Given these research seeds, perform deep analysis:

1. **Claim Verification**: For each core claim, argue BOTH for and against it. Identify which side has stronger evidence.
2. **Blind Spot Detection**: What perspectives are completely missing from the original analysis?
3. **Assumption Audit**: List every implicit assumption in the original content.
4. **Alternative Explanations**: For each major finding, propose at least 2 alternative explanations.
5. **Risk Assessment**: What are the risks of being wrong about the key claims?
6. **Synthesis**: Reconcile the adversarial findings into a nuanced conclusion.

RESEARCH SEEDS:
{seed_doc[:8000]}"""

    try:
        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "system", "content": "You are an adversarial research analyst. Challenge assumptions, find blind spots, and stress-test claims. Output in clean markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.4,
        )
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", "") or ""
        return {"status": "success", "content": content.strip()}
    except Exception as e:
        return {"status": "error", "content": f"Deep research failed: {e}"}


# ─── Pipeline: Circular Dependencies Agent ───────────────────────────────────

def run_circular_dependencies(text: str) -> dict:
    """Clean and structure text using the Circular Dependencies Agent."""
    try:
        resp = httpx.post(
            "https://ax-opencode-translator.vercel.app/api/translate",
            json={
                "text": (
                    "Convert telegraphic notes into a structured document "
                    "with circular dependencies removal. Preserve Facts, "
                    "headings, subheadings, bullet points. Add "
                    "Argumentative connectives and logical flow. "
                    "Style polished.\n\nInput:\n\n" + text
                ),
                "sourceLanguage": "en",
                "targetLanguage": "en",
                "fast": True
            },
            timeout=60
        )
        if resp.status_code == 200:
            cleaned = resp.json().get("translatedText", "")
            return {"status": "success", "content": cleaned}
        else:
            return {"status": "error", "content": f"Circular Dependencies returned: {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "content": f"Circular Dependencies failed: {e}"}


# ─── Pipeline: Atomic Graph ──────────────────────────────────────────────────

def run_atomic_graph(text: str) -> dict:
    """Build a knowledge graph from text using the Atomic Graph pipeline."""
    system_prompt = (
        "You are a semantic reasoning engine that builds knowledge graphs "
        "from raw thinking. You do NOT merely reformat or summarise — you "
        "REASON through the semantic space of ideas. CRITICAL OUTPUT RULES: "
        "Always respond with valid JSON only. No markdown, no explanation, "
        "no code fences."
    )

    extract_prompt = (
        "Extract atomic concepts from these notes. Each concept = ONE idea only. "
        "Return JSON: "
        '{ "nodes": [{ "id": "c1", "title": "...", '
        '"summary": "...", "tags": ["..."] }], '
        '"edges": [{ "source": "c1", "target": "c2", '
        '"relation": "...", "weight": 0.9 }] }\n\n'
        f"Raw notes:\n{text[:5000]}"
    )

    try:
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
            try:
                graph = json.loads(content)
                return {"status": "success", "content": json.dumps(graph, indent=2, ensure_ascii=False)}
            except json.JSONDecodeError:
                return {"status": "success", "content": content}
        else:
            return {"status": "error", "content": f"Atomic Graph returned: {ag_resp.status_code}"}
    except Exception as e:
        return {"status": "error", "content": f"Atomic Graph failed: {e}"}


# ─── Pipeline: RAG Second Brain ──────────────────────────────────────────────

def save_to_rag(text: str, name: str = None) -> dict:
    """Save text to the RAG Document Assistant (Second Brain)."""
    if name is None:
        name = f"transcript_{int(time.time())}.txt"

    try:
        rag_resp = httpx.post(
            "https://rag-document-assistant-three.vercel.app/api/upload",
            json={
                "type": "text",
                "content": text,
                "name": name,
                "mode": "Add"
            },
            timeout=30
        )
        if rag_resp.status_code == 200:
            return {"status": "success", "content": "Saved to Second Brain!"}
        else:
            return {"status": "error", "content": f"RAG returned: {rag_resp.status_code}"}
    except Exception as e:
        return {"status": "error", "content": f"RAG save failed: {e}"}


def query_rag(query: str) -> dict:
    """Query the RAG Document Assistant."""
    try:
        rag_resp = httpx.post(
            "https://rag-document-assistant-three.vercel.app/api/query",
            json={"query": query},
            timeout=30
        )
        if rag_resp.status_code == 200:
            data = rag_resp.json()
            return {"status": "success", "content": data.get("response", str(data))}
        else:
            return {"status": "error", "content": f"RAG query returned: {rag_resp.status_code}"}
    except Exception as e:
        return {"status": "error", "content": f"RAG query failed: {e}"}


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Audio Pipeline | MiMo V2.5 ASR",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "seed_research" not in st.session_state:
        st.session_state.seed_research = ""
    if "deep_research" not in st.session_state:
        st.session_state.deep_research = ""
    if "cleaned_text" not in st.session_state:
        st.session_state.cleaned_text = ""
    if "atomic_graph" not in st.session_state:
        st.session_state.atomic_graph = ""

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
        st.markdown("### 📋 Pipeline Stages")
        st.markdown("""
        1. **🎤 Transcribe** — Audio → Text (MiMo V2.5)
        2. **🌱 Seed Research** — Extract topics & claims
        3. **🔬 Deep Research** — Adversarial analysis
        4. **🔄 Circular Deps** — Structure & clean
        5. **🧬 Atomic Graph** — Knowledge map
        6. **🧠 RAG** — Second brain storage
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
        No pydub — uses ffmpeg (Python 3.13+ safe)<br/>
        Supports: MP3, WAV, M4A, OGG, WEBM, FLAC
        </small>
        """, unsafe_allow_html=True)

    # ── Main Area ────────────────────────────────────────────────────────────
    st.title("🎙️ Audio → Knowledge Pipeline")
    st.caption("Transcribe, research, structure, and store — powered by MiMo V2.5 ASR")

    # ── Step 1: Audio Input ──────────────────────────────────────────────────
    st.markdown("## 🎤 Step 1: Audio Input")

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

        # Cookie upload for YouTube authentication
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
        audio_value = st.audio_input(
            "Click to record",
            key="audio_record",
            help="Click the microphone button to start recording"
        )
        if audio_value:
            raw_audio_path = save_uploaded_file(audio_value)
            source_label = "Recording"
            st.audio(audio_value)

    # ── Convert & Show Info ──────────────────────────────────────────────────
    st.markdown("---")

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
        st.markdown("## 🚀 Step 2: Transcribe")

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
            height=300,
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
                    "duration_minutes": round(duration_min, 2) if wav_path else 0,
                    "language": language,
                    "model": model_choice,
                    "transcript": edited_transcript,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, indent=2, ensure_ascii=False),
                file_name="transcript.json",
                mime="application/json",
                key="download_json"
            )

        # ── Step 3: Seed Research ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🌱 Step 3: Seed Research")

        if st.button("🌱 Run Seed Research", key="seed_btn"):
            with st.spinner("Extracting topics and claims from transcript..."):
                result = run_seed_research(edited_transcript, api_key)
                if result["status"] == "success":
                    st.session_state.seed_research = result["content"]
                    st.success("✅ Seed research complete!")
                else:
                    st.error(f"❌ {result['content']}")

        if st.session_state.seed_research:
            seed_text = st.text_area(
                "Seed Research (editable)",
                value=st.session_state.seed_research,
                height=300,
                key="seed_output"
            )
            st.session_state.seed_research = seed_text
            st.download_button(
                "📄 Download seed_research.md",
                data=seed_text.encode("utf-8"),
                file_name="seed_research.md",
                mime="text/markdown",
                key="download_seed"
            )

            # ── Step 4: Deep Research ────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 🔬 Step 4: Deep Research (Adversarial)")

            if st.button("🔬 Run Deep Research", key="deep_btn"):
                with st.spinner("Running adversarial analysis on seed research..."):
                    result = run_deep_research(seed_text, api_key)
                    if result["status"] == "success":
                        st.session_state.deep_research = result["content"]
                        st.success("✅ Deep research complete!")
                    else:
                        st.error(f"❌ {result['content']}")

            if st.session_state.deep_research:
                deep_text = st.text_area(
                    "Deep Research (editable)",
                    value=st.session_state.deep_research,
                    height=300,
                    key="deep_output"
                )
                st.session_state.deep_research = deep_text
                st.download_button(
                    "📄 Download deep_research.md",
                    data=deep_text.encode("utf-8"),
                    file_name="deep_research.md",
                    mime="text/markdown",
                    key="download_deep"
                )

        # ── Step 5: Circular Dependencies ────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔄 Step 5: Circular Dependencies Cleanup")

        cleanup_input = st.radio(
            "Input for cleanup:",
            ["Transcript", "Seed Research", "Deep Research"],
            key="cd_input_choice",
            horizontal=True
        )
        cd_input_text = {
            "Transcript": st.session_state.transcript,
            "Seed Research": st.session_state.seed_research,
            "Deep Research": st.session_state.deep_research,
        }.get(cleanup_input, st.session_state.transcript)

        if st.button("🔄 Run Circular Dependencies Cleanup", key="cd_btn",
                     disabled=not cd_input_text):
            with st.spinner("Cleaning and structuring text..."):
                result = run_circular_dependencies(cd_input_text)
                if result["status"] == "success":
                    st.session_state.cleaned_text = result["content"]
                    st.success("✅ Cleanup complete!")
                else:
                    st.error(f"❌ {result['content']}")

        if st.session_state.cleaned_text:
            cleaned_text = st.text_area(
                "Cleaned Text (editable)",
                value=st.session_state.cleaned_text,
                height=300,
                key="cd_output"
            )
            st.session_state.cleaned_text = cleaned_text
            st.download_button(
                "📄 Download cleaned.txt",
                data=cleaned_text.encode("utf-8"),
                file_name="transcript_cleaned.txt",
                mime="text/plain",
                key="download_cleaned"
            )

        # ── Step 6: Atomic Graph ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🧬 Step 6: Atomic Graph (Knowledge Map)")

        graph_input = st.radio(
            "Input for knowledge graph:",
            ["Transcript", "Seed Research", "Deep Research", "Cleaned Text"],
            key="ag_input_choice",
            horizontal=True
        )
        ag_input_text = {
            "Transcript": st.session_state.transcript,
            "Seed Research": st.session_state.seed_research,
            "Deep Research": st.session_state.deep_research,
            "Cleaned Text": st.session_state.cleaned_text,
        }.get(graph_input, st.session_state.transcript)

        if st.button("🧬 Build Knowledge Graph", key="ag_btn",
                     disabled=not ag_input_text):
            with st.spinner("Building knowledge graph..."):
                result = run_atomic_graph(ag_input_text)
                if result["status"] == "success":
                    st.session_state.atomic_graph = result["content"]
                    st.success("✅ Knowledge graph built!")
                else:
                    st.error(f"❌ {result['content']}")

        if st.session_state.atomic_graph:
            graph_text = st.text_area(
                "Knowledge Graph JSON",
                value=st.session_state.atomic_graph,
                height=300,
                key="ag_output"
            )
            st.session_state.atomic_graph = graph_text
            st.download_button(
                "📄 Download atomic_graph.json",
                data=graph_text.encode("utf-8"),
                file_name="atomic_graph.json",
                mime="application/json",
                key="download_graph"
            )

        # ── Step 7: RAG Second Brain ────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🧠 Step 7: Second Brain (RAG)")

        rag_input = st.radio(
            "Document to store:",
            ["Transcript", "Seed Research", "Deep Research", "Cleaned Text", "All Combined"],
            key="rag_input_choice",
            horizontal=True
        )

        rag_input_text = {
            "Transcript": st.session_state.transcript,
            "Seed Research": st.session_state.seed_research,
            "Deep Research": st.session_state.deep_research,
            "Cleaned Text": st.session_state.cleaned_text,
            "All Combined": "\n\n---\n\n".join(filter(None, [
                f"# TRANSCRIPT\n{st.session_state.transcript}",
                f"# SEED RESEARCH\n{st.session_state.seed_research}",
                f"# DEEP RESEARCH\n{st.session_state.deep_research}",
                f"# CLEANED TEXT\n{st.session_state.cleaned_text}",
            ])),
        }.get(rag_input, st.session_state.transcript)

        col_rag1, col_rag2 = st.columns(2)

        with col_rag1:
            if st.button("🧠 Save to Second Brain", key="rag_save_btn",
                         disabled=not rag_input_text):
                with st.spinner("Saving to Second Brain..."):
                    result = save_to_rag(rag_input_text)
                    if result["status"] == "success":
                        st.success(f"✅ {result['content']}")
                    else:
                        st.error(f"❌ {result['content']}")

        with col_rag2:
            rag_query = st.text_input(
                "Query Second Brain:",
                placeholder="Ask a question about your stored documents...",
                key="rag_query"
            )
            if st.button("🔍 Query", key="rag_query_btn", disabled=not rag_query):
                with st.spinner("Querying Second Brain..."):
                    result = query_rag(rag_query)
                    if result["status"] == "success":
                        st.markdown("### 📖 Answer")
                        st.write(result["content"])
                    else:
                        st.error(f"❌ {result['content']}")

    else:
        st.info("👆 Upload an audio file, enter a YouTube URL, or record audio to get started.")


if __name__ == "__main__":
    main()
