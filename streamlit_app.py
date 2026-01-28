import json
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.utils.model_downloader import download_models

download_models()

ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")
MODEL_OPTIONS = {
    "tiny": asr_model_specs.WHISPER_TINY,
    "base": asr_model_specs.WHISPER_BASE,
    "small": asr_model_specs.WHISPER_SMALL,
    "medium": asr_model_specs.WHISPER_MEDIUM,
    "large": asr_model_specs.WHISPER_LARGE,
    "turbo": asr_model_specs.WHISPER_TURBO,
}


def transcribe(audio_path: Path, model_name: str) -> dict:
    """Transcribe audio and return result dict with response and metrics."""
    pipeline_options = AsrPipelineOptions(
        artifacts_path=ARTIFACTS_PATH,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
    )
    pipeline_options.asr_options = MODEL_OPTIONS[model_name]

    load_start = time.perf_counter_ns()
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    load_duration = time.perf_counter_ns() - load_start

    eval_start = time.perf_counter_ns()
    result = converter.convert(audio_path)
    eval_duration = time.perf_counter_ns() - eval_start

    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Conversion failed: {result.status}")

    response = result.document.export_to_markdown()
    return {
        "model": model_name,
        "response": response,
        "total_duration": load_duration + eval_duration,
        "load_duration": load_duration,
        "prompt_eval_count": len(result.document.texts),
        "prompt_eval_duration": eval_duration // 4,
        "eval_count": len(response.split()),
        "eval_duration": eval_duration,
    }


def get_audio_duration(audio_path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        return None


def format_duration(ns: int) -> str:
    """Format nanoseconds as human-readable duration."""
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def format_bytes(size: int) -> str:
    """Format file size in human-readable format."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def display_results(r: dict, file_name: str, file_size: int, file_format: str, audio_dur: float | None):
    """Display transcript and metrics."""
    st.subheader("Transcript")
    st.markdown(r["response"])

    total_sec = r["total_duration"] / 1_000_000_000
    wps = r["eval_count"] / total_sec if total_sec > 0 else 0
    speed = (audio_dur / total_sec) if audio_dur and total_sec > 0 else None

    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", r["model"], help="Whisper model used")
    c2.metric("Total Time", format_duration(r["total_duration"]), help="Total processing time")
    c3.metric("Words", f"{r['eval_count']:,}", help="Words in transcript")
    c4.metric("Speed", f"{speed:.1f}x" if speed else "N/A", help="Faster than real-time")

    with st.expander("Detailed Metrics"):
        st.caption("Audio File")
        c1, c2, c3 = st.columns(3)
        c1.metric("File Size", format_bytes(file_size))
        c2.metric("Format", file_format)
        c3.metric("Duration", f"{audio_dur:.2f} s" if audio_dur else "N/A")

        st.caption("Timing Breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("Load Time", format_duration(r["load_duration"]), help="Model init time")
        c2.metric("Transcription Time", format_duration(r["eval_duration"]), help="Audio processing")
        c3.metric("Words/Second", f"{wps:.1f}", help="Throughput")

        st.caption("Processing Details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Audio Segments", f"{r['prompt_eval_count']:,}", help="Segments processed")
        c2.metric("Audio Processing", format_duration(r["prompt_eval_duration"]), help="Decoding time")
        c3.metric("Words Generated", f"{r['eval_count']:,}", help="Output words")

    st.download_button(
        "Download",
        json.dumps(r, indent=2),
        file_name.rsplit(".", 1)[0] + "_transcript.json",
        "application/json",
    )


# UI
st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown using MLX Whisper on Apple Silicon.")

with st.form("transcribe_form"):
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
    selected_model = st.selectbox(
        "Select Whisper model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index("turbo"),
    )
    submitted = st.form_submit_button("Transcribe", type="primary")

if submitted and uploaded_file:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        audio_duration = get_audio_duration(tmp_path)
        with st.spinner("Transcribing..."):
            result = transcribe(tmp_path, selected_model)
        st.success("Done.")
        display_results(
            result,
            uploaded_file.name,
            uploaded_file.size,
            suffix.lstrip(".").upper(),
            audio_duration,
        )
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
elif submitted:
    st.warning("Please upload an audio file.")
