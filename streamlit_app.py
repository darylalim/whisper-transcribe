import json
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.datamodel import asr_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

ASR_MODEL = asr_model_specs.WHISPER_TURBO_MLX
AUDIO_FORMATS = ("wav", "mp3", "m4a", "ogg", "flac", "webm", "aac")
ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")


@st.cache_resource
def _get_converter() -> DocumentConverter:
    options = AsrPipelineOptions(
        artifacts_path=ARTIFACTS_PATH,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
    )
    options.asr_options = ASR_MODEL
    return DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(pipeline_cls=AsrPipeline, pipeline_options=options)
        }
    )


def _get_audio_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _transcribe(path: Path) -> tuple[str, float]:
    converter = _get_converter()
    start = time.perf_counter()
    result = converter.convert(path)
    elapsed = round(time.perf_counter() - start, 2)
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Conversion failed: {result.status}")
    return result.document.export_to_markdown(), elapsed


def _handle_transcription(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        audio_duration = _get_audio_duration(tmp_path)
        if audio_duration is None:
            st.warning("Could not determine audio duration. Transcribing anyway.")

        with st.spinner("Transcribing..."):
            transcript, eval_duration = _transcribe(tmp_path)

        num_words = len(transcript.split())
        file_stem = Path(uploaded_file.name).stem + "_transcript"

        parts: list[str] = []
        if audio_duration is not None:
            parts.append(f"{audio_duration:.1f}s audio")
        parts.append(f"{num_words:,} words")
        parts.append(f"transcribed in {eval_duration:.2f}s")
        st.caption(" · ".join(parts))

        st.text_area(
            "Transcript", transcript, height=300, disabled=True, label_visibility="collapsed"
        )

        c1, c2 = st.columns(2)
        c1.download_button("Download transcript", transcript, file_stem + ".txt", "text/plain")
        c2.download_button(
            "Download JSON",
            json.dumps(
                {
                    "audio_duration": audio_duration,
                    "transcript": transcript,
                    "num_words": num_words,
                    "eval_duration": eval_duration,
                },
                indent=2,
            ),
            file_stem + ".json",
            "application/json",
        )
    except RuntimeError as e:
        st.error(f"Transcription failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.exception(e)
    finally:
        tmp_path.unlink(missing_ok=True)


# UI
st.title("Audio Transcription")
st.write("Record or upload audio to transcribe with Whisper.")

recorded_audio = st.audio_input("Record audio")
if recorded_audio:
    st.audio(recorded_audio)
record_submitted = st.button(
    "Transcribe", type="primary", key="record_btn", disabled=not recorded_audio
)

st.divider()

uploaded_file = st.file_uploader("Upload audio file", type=AUDIO_FORMATS)
if uploaded_file:
    st.audio(uploaded_file)
upload_submitted = st.button(
    "Transcribe", type="primary", key="upload_btn", disabled=not uploaded_file
)

if record_submitted and recorded_audio:
    _handle_transcription(recorded_audio)
elif upload_submitted and uploaded_file:
    _handle_transcription(uploaded_file)
