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


# UI
st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown using MLX Whisper on Apple Silicon.")

with st.form("transcribe_form"):
    uploaded_file = st.file_uploader("Upload audio file", type=AUDIO_FORMATS)
    submitted = st.form_submit_button("Transcribe", type="primary")

if submitted and uploaded_file:
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
        st.success("Done.")

        st.subheader("Transcript")
        st.markdown(transcript)

        num_words = len(transcript.split())

        st.subheader("Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Audio Duration", f"{audio_duration:.2f} s" if audio_duration else "N/A")
        c2.metric("Words", f"{num_words:,}")
        c3.metric("Eval Duration", f"{eval_duration:.2f} s")

        st.download_button(
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
            Path(uploaded_file.name).stem + "_transcript.json",
            "application/json",
        )
    except RuntimeError as e:
        st.error(f"Transcription failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.exception(e)
    finally:
        tmp_path.unlink(missing_ok=True)
elif submitted:
    st.warning("Please upload an audio file.")
