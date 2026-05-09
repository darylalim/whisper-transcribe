import tempfile
from pathlib import Path

import mlx_whisper
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
AUDIO_FORMATS = ("aac", "flac", "m4a", "mov", "mp3", "mp4", "ogg", "wav", "webm")


@st.cache_data(show_spinner="Transcribing...")
def _transcribe(audio_bytes: bytes, suffix: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=ASR_MODEL_REPO,
            language="en",
            task="transcribe",
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result


def _handle_transcription(uploaded_file: UploadedFile) -> None:
    name = Path(uploaded_file.name)
    try:
        result = _transcribe(uploaded_file.read(), name.suffix)
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
    _handle_transcription(audio_source)

_display_transcription()
