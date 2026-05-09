import tempfile
from pathlib import Path

import mlx_whisper
import streamlit as st
from mlx_whisper.tokenizer import LANGUAGES
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
AUDIO_FORMATS = ("aac", "flac", "m4a", "mov", "mp3", "mp4", "ogg", "wav", "webm")
LANGUAGE_CODES: list[str | None] = [None] + sorted(LANGUAGES, key=lambda c: LANGUAGES[c])


def _format_language(code: str | None) -> str:
    return "Detect" if code is None else LANGUAGES[code].title()


@st.cache_data(show_spinner="Transcribing...")
def _transcribe(
    audio_bytes: bytes, suffix: str, language: str | None = None, task: str = "transcribe"
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=ASR_MODEL_REPO,
            language=language,
            task=task,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result


def _handle_transcription(uploaded_file: UploadedFile, language: str | None, task: str) -> None:
    name = Path(uploaded_file.name)
    try:
        result = _transcribe(uploaded_file.read(), name.suffix, language, task)
        st.session_state["transcription"] = {
            "result": result,
            "file_stem": name.stem + "_transcript",
        }
    except RuntimeError as e:
        st.error(f"Transcription failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.exception(e)


def _display_transcription() -> None:
    if (data := st.session_state.get("transcription")) is None:
        return
    transcript = data["result"]["text"].strip()
    st.text_area("Transcript", transcript, height=300, disabled=True, label_visibility="collapsed")
    st.download_button("Download", transcript, data["file_stem"] + ".txt", "text/plain")


# UI
st.title("Whisper Pipeline")

upload_tab, record_tab = st.tabs(["Upload", "Record"])
with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload audio file", type=AUDIO_FORMATS, label_visibility="collapsed"
    )
    if uploaded_file:
        st.audio(uploaded_file)

with record_tab:
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    if recorded_audio:
        st.audio(recorded_audio)

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

audio_source = uploaded_file or recorded_audio
_, action_col = st.columns([3, 1])
with action_col:
    transcribe_clicked = st.button(
        "Transcribe",
        type="primary",
        disabled=audio_source is None,
        use_container_width=True,
    )

if transcribe_clicked and audio_source is not None:
    _handle_transcription(audio_source, language, "translate" if translate else "transcribe")

_display_transcription()
