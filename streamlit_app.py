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

MODEL_OPTIONS = {
    "tiny": asr_model_specs.WHISPER_TINY_MLX,
    "base": asr_model_specs.WHISPER_BASE_MLX,
    "small": asr_model_specs.WHISPER_SMALL_MLX,
    "medium": asr_model_specs.WHISPER_MEDIUM_MLX,
    "large": asr_model_specs.WHISPER_LARGE_MLX,
    "turbo": asr_model_specs.WHISPER_TURBO_MLX,
}

AUDIO_FORMATS = ("wav", "mp3", "m4a", "ogg", "flac", "webm", "aac")
MODEL_NAMES = list(MODEL_OPTIONS.keys())


@st.cache_resource
def _get_converter(model_name: str) -> DocumentConverter:
    """Create and cache a DocumentConverter for the given model."""
    options = AsrPipelineOptions(
        artifacts_path=str(Path.home() / ".cache" / "docling" / "models"),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
    )
    options.asr_options = MODEL_OPTIONS[model_name]
    return DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=options,
            )
        }
    )


def get_audio_duration(audio_path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe."""
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
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def transcribe(audio_path: Path, model_name: str, audio_duration: float | None) -> dict:
    """Transcribe audio and return result dict with transcript and metrics."""
    converter = _get_converter(model_name)

    eval_start = time.perf_counter()
    result = converter.convert(audio_path)
    eval_duration = time.perf_counter() - eval_start

    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Conversion failed: {result.status}")

    transcript = result.document.export_to_markdown()
    return {
        "model": model_name,
        "audio_duration": audio_duration,
        "transcript": transcript,
        "num_words": len(transcript.split()),
        "eval_duration": round(eval_duration, 2),
    }


# UI
st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown using MLX Whisper on Apple Silicon.")

with st.form("transcribe_form"):
    uploaded_file = st.file_uploader("Upload audio file", type=AUDIO_FORMATS)
    selected_model = st.selectbox(
        "Select Whisper model",
        MODEL_NAMES,
        index=MODEL_NAMES.index("turbo"),
    )
    submitted = st.form_submit_button("Transcribe", type="primary")

if submitted and uploaded_file:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        audio_duration = get_audio_duration(tmp_path)
        if audio_duration is None:
            st.warning("Could not determine audio duration. Transcribing anyway.")

        with st.spinner("Transcribing..."):
            r = transcribe(tmp_path, selected_model, audio_duration)
        st.success("Done.")

        st.subheader("Transcript")
        st.markdown(r["transcript"])

        st.subheader("Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", r["model"])
        dur = f"{r['audio_duration']:.2f} s" if r["audio_duration"] else "N/A"
        c2.metric("Audio Duration", dur)
        c3.metric("Words", f"{r['num_words']:,}")
        c4.metric("Eval Duration", f"{r['eval_duration']:.2f} s")

        st.download_button(
            "Download JSON",
            json.dumps(r, indent=2),
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
