import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlx_whisper
import streamlit as st
import yt_dlp
from mlx_whisper.tokenizer import LANGUAGES
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
AUDIO_FORMATS = ("mp3", "m4a", "wav", "flac", "ogg", "aac", "mp4", "mov", "webm", "mkv")
LANGUAGE_CODES: list[str | None] = [None] + sorted(LANGUAGES, key=lambda c: LANGUAGES[c])
YOUTUBE_URL_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE)


class _YouTubeAudio:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data


@st.cache_data(show_spinner="Downloading audio from YouTube...", max_entries=5)
def _fetch_youtube_audio(url: str) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmpdir) / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "restrictfilenames": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
        return downloaded.read_bytes(), downloaded.name


def _format_language(code: str | None) -> str:
    return "Detect" if code is None else LANGUAGES[code].title()


def _format_timestamp(seconds: float, decimal_marker: str = ".") -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def _format_srt(result: dict) -> str:
    return "\n".join(
        f"{i}\n"
        f"{_format_timestamp(s['start'], decimal_marker=',')} --> "
        f"{_format_timestamp(s['end'], decimal_marker=',')}\n"
        f"{s['text'].strip().replace('-->', '->')}\n"
        for i, s in enumerate(result["segments"], start=1)
    )


@st.cache_data(show_spinner=False, max_entries=20)
def _transcribe(
    audio_bytes: bytes,
    suffix: str,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=ASR_MODEL_REPO,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=no_verbatim,
            hallucination_silence_threshold=2.0 if no_verbatim else None,
        )
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result


def _handle_transcription(
    uploaded_files: Sequence[UploadedFile | _YouTubeAudio],
    language: str | None,
    task: str,
    include_subtitles: bool,
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
) -> None:
    transcriptions = []
    total = len(uploaded_files)
    with st.status(f"Transcribing {total} file(s)...", expanded=True) as status:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status.update(label=f"Transcribing {uploaded_file.name} ({i}/{total})...")
            name = Path(uploaded_file.name)
            try:
                result = _transcribe(
                    uploaded_file.read(),
                    name.suffix,
                    language,
                    task,
                    initial_prompt,
                    no_verbatim,
                )
                transcriptions.append(
                    {
                        "result": result,
                        "file_stem": f"{name.stem}_{name.suffix.lstrip('.')}_transcript",
                        "filename": uploaded_file.name,
                        "include_subtitles": include_subtitles,
                    }
                )
            except RuntimeError as e:
                st.error(f"Transcription failed for {uploaded_file.name}: {e}")
            except Exception as e:
                st.error(f"Unexpected error for {uploaded_file.name}: {e}")
                st.exception(e)
        status.update(
            label=f"Transcribed {len(transcriptions)}/{total} file(s)",
            state="complete",
        )
    st.session_state["transcription"] = transcriptions


def _display_transcription() -> None:
    transcriptions = st.session_state.get("transcription") or []
    for i, data in enumerate(transcriptions):
        include_subtitles = data["include_subtitles"]
        if include_subtitles:
            initial = _format_srt(data["result"])
        else:
            initial = data["result"]["text"].strip()
        st.subheader(data["filename"])
        transcript = st.text_area(
            "Transcript",
            initial,
            height=300,
            label_visibility="collapsed",
            key=f"transcript_{i}",
        )
        if include_subtitles:
            st.download_button(
                ".srt",
                transcript,
                data["file_stem"] + ".srt",
                "application/x-subrip",
                key=f"download_srt_{i}",
                use_container_width=True,
            )
        else:
            st.download_button(
                ".txt",
                transcript,
                data["file_stem"] + ".txt",
                "text/plain",
                key=f"download_txt_{i}",
                use_container_width=True,
            )


# UI
st.title("Whisper Pipeline")

upload_tab, record_tab, youtube_tab = st.tabs(["Upload", "Record", "YouTube"])
with upload_tab:
    uploaded_files = st.file_uploader(
        "Upload audio file",
        type=AUDIO_FORMATS,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file)

with record_tab:
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    if recorded_audio:
        st.audio(recorded_audio)

with youtube_tab:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    ).strip()
    youtube_audio: _YouTubeAudio | None = None
    if youtube_url and YOUTUBE_URL_RE.match(youtube_url):
        try:
            data, filename = _fetch_youtube_audio(youtube_url)
            youtube_audio = _YouTubeAudio(filename, data)
            st.audio(data)
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Could not download from YouTube: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.exception(e)

language_label_col, language_col = st.columns([3, 1], vertical_alignment="center")
with language_label_col:
    st.markdown(
        "Primary language",
        help=(
            "The primary language spoken in an uploaded file. "
            "By default, the primary language will be detected automatically."
        ),
    )
with language_col:
    language = st.selectbox(
        "Primary language",
        LANGUAGE_CODES,
        format_func=_format_language,
        label_visibility="collapsed",
    )

translate_label_col, translate_col = st.columns([3, 1], vertical_alignment="center")
with translate_label_col:
    st.markdown(
        "Translate to English",
        help="Translates audio to English instead of transcribing in the source language.",
    )
with translate_col:
    with st.container(horizontal_alignment="right"):
        translate = st.toggle("Translate to English", value=False, label_visibility="collapsed")

subtitles_label_col, subtitles_col = st.columns([3, 1], vertical_alignment="center")
with subtitles_label_col:
    st.markdown(
        "Include subtitles",
        help=(
            "Best for adding subtitles to a video. When enabled, the project will be "
            "initialized with subtitles which you can then alter in the editor."
        ),
    )
with subtitles_col:
    with st.container(horizontal_alignment="right"):
        include_subtitles = st.toggle(
            "Include subtitles", value=False, label_visibility="collapsed"
        )

no_verbatim_label_col, no_verbatim_col = st.columns([3, 1], vertical_alignment="center")
with no_verbatim_label_col:
    st.markdown(
        "No verbatim",
        help=(
            "When enabled, the transcription will be cleaned up by removing "
            "filler words, false starts, and repetitions."
        ),
    )
with no_verbatim_col:
    with st.container(horizontal_alignment="right"):
        no_verbatim = st.toggle("No verbatim", value=False, label_visibility="collapsed")

keyterms_label_col, _ = st.columns([3, 1], vertical_alignment="center")
with keyterms_label_col:
    st.markdown(
        "Keyterms",
        help=(
            "Up to 50 keyterms to be boosted during transcription. "
            "Boosted terms are more likely to appear in the output."
        ),
    )
keyterms = st.multiselect(
    "Keyterms",
    options=[],
    accept_new_options=True,
    max_selections=50,
    placeholder="Add keyterms...",
    label_visibility="collapsed",
)
initial_prompt = ", ".join(keyterms) or None

audio_sources = (
    uploaded_files
    or ([recorded_audio] if recorded_audio else [])
    or ([youtube_audio] if youtube_audio else [])
)
_, action_col = st.columns([3, 1])
with action_col:
    transcribe_clicked = st.button(
        "Transcribe",
        type="primary",
        disabled=not audio_sources,
        use_container_width=True,
    )

if transcribe_clicked and audio_sources:
    _handle_transcription(
        audio_sources,
        language,
        "translate" if translate else "transcribe",
        include_subtitles,
        initial_prompt,
        no_verbatim,
    )

_display_transcription()
