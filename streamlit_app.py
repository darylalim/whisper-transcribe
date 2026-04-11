import tempfile
from pathlib import Path

import mlx_whisper
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
AUDIO_FORMATS = ("aac", "flac", "m4a", "mov", "mp3", "mp4", "ogg", "wav", "webm")


def _transcribe(path: Path) -> dict:
    result = mlx_whisper.transcribe(
        str(path),
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"audio{name.suffix}"
        tmp_path.write_bytes(uploaded_file.read())

        try:
            with st.spinner("Transcribing..."):
                result = _transcribe(tmp_path)

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
    if "transcription" not in st.session_state:
        return

    data = st.session_state["transcription"]
    result = data["result"]
    file_stem = data["file_stem"]

    transcript = result["text"].strip()

    st.text_area("Transcript", transcript, height=300, disabled=True, label_visibility="collapsed")

    st.download_button("Download", transcript, file_stem + ".txt", "text/plain")


# UI
st.title("Whisper Pipeline")
record_tab, upload_tab = st.tabs(["Record", "Upload"])
with record_tab:
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    if recorded_audio:
        st.audio(recorded_audio)
    record_submitted = st.button(
        "Transcribe", type="primary", key="record_btn", disabled=not recorded_audio
    )

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload audio file", type=AUDIO_FORMATS, label_visibility="collapsed"
    )
    if uploaded_file:
        st.audio(uploaded_file)
    upload_submitted = st.button(
        "Transcribe", type="primary", key="upload_btn", disabled=not uploaded_file
    )

if record_submitted and recorded_audio:
    _handle_transcription(recorded_audio)
elif upload_submitted and uploaded_file:
    _handle_transcription(uploaded_file)

_display_transcription()
