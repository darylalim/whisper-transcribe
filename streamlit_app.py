import json
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling_core.types.doc import DoclingDocument
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.utils.model_downloader import download_models

# Docling models can be prefetched for offline use
download_models()

artifacts_path = str(Path.home() / '.cache' / 'docling' / 'models')

MODEL_OPTIONS = {
    "tiny": asr_model_specs.WHISPER_TINY,
    "base": asr_model_specs.WHISPER_BASE,
    "small": asr_model_specs.WHISPER_SMALL,
    "medium": asr_model_specs.WHISPER_MEDIUM,
    "large": asr_model_specs.WHISPER_LARGE,
    "turbo": asr_model_specs.WHISPER_TURBO,
}

def get_asr_converter(model_name: str = "turbo"):
    """Create a DocumentConverter configured for ASR with the specified model.

    Args:
        model_name: Name of the Whisper model to use. Options are:
            tiny, base, small, medium, large, turbo
    
    The selected model automatically uses the best implementation for your hardware:
    - MLX Whisper for Apple Silicon (M1/M2/M3) with mlx-whisper installed
    - Native Whisper as fallback
    """
    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
    
    pipeline_options = AsrPipelineOptions(
        artifacts_path=artifacts_path,
        accelerator_options=accelerator_options
    )
    pipeline_options.asr_options = MODEL_OPTIONS[model_name]

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter

def asr_pipeline_conversion(audio_path: Path, model_name: str = "turbo") -> tuple[DoclingDocument, dict]:
    """Run the ASR pipeline and return a `DoclingDocument` transcript with metrics.
    
    Returns:
        tuple: (DoclingDocument, metrics_dict) where metrics_dict contains:
            - model: Name of the model used
            - total_duration_ns: Total transcription time in nanoseconds
            - segment_count: Number of transcript segments
    """
    assert audio_path.exists(), f"Audio file not found: {audio_path}"

    # Measure total conversion time
    start_time_ns = time.perf_counter_ns()
    
    converter = get_asr_converter(model_name)

    # Convert the audio file
    result: ConversionResult = converter.convert(audio_path)
    
    end_time_ns = time.perf_counter_ns()
    total_duration_ns = end_time_ns - start_time_ns

    # Verify conversion was successful
    assert result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {result.status}"
    )
    
    metrics = {
        "model": model_name,
        "total_duration_ns": total_duration_ns,
        "segment_count": len(result.document.texts),
    }
    
    return result.document, metrics

def format_duration(nanoseconds: int) -> str:
    """Format nanoseconds with thousand separators."""
    return f"{nanoseconds:,} ns"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_audio_duration(audio_path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe.
    
    Returns:
        Duration in seconds, or None if ffprobe fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        return None

st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown text using the MLX Whisper models on Apple Silicon devices.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3"],
    accept_multiple_files=False,
    help="Accepts .wav and .mp3 files."
)

selected_model = st.selectbox(
    "Select Whisper model",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index("turbo"),
)

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    
    if st.button("Transcribe", type="primary"):
        # Save uploaded file to temporary location with correct extension
        file_suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)
        
        try:
            file_size = uploaded_file.size
            audio_format = file_suffix.lstrip(".").upper()
            audio_duration = get_audio_duration(tmp_path)
            
            with st.spinner("Transcribing..."):
                doc, metrics = asr_pipeline_conversion(audio_path=tmp_path, model_name=selected_model)
                transcript = doc.export_to_markdown()
            
            # Calculate output metrics
            word_count = len(transcript.split())
            character_count = len(transcript)
            
            # Calculate real-time factor (audio duration / transcription time)
            if audio_duration is not None:
                transcription_seconds = metrics["total_duration_ns"] / 1_000_000_000
                real_time_factor = transcription_seconds / audio_duration
            else:
                real_time_factor = None
            
            metrics["file_size"] = file_size
            metrics["audio_format"] = audio_format
            metrics["audio_duration"] = audio_duration
            metrics["word_count"] = word_count
            metrics["character_count"] = character_count
            metrics["real_time_factor"] = real_time_factor
            
            st.session_state.transcript = transcript
            st.session_state.filename = uploaded_file.name
            st.session_state.metrics = metrics
            
            st.success("Done.")
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

if "transcript" in st.session_state:
    st.subheader("Transcript")
    st.markdown(st.session_state.transcript)
    
    if "metrics" in st.session_state:
        st.subheader("Metrics")
        metrics = st.session_state.metrics
        
        # Audio file metrics
        st.caption("Audio File")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="File Size", value=format_file_size(metrics["file_size"]))
        with col2:
            st.metric(label="Audio Format", value=metrics["audio_format"])
        with col3:
            if metrics["audio_duration"] is not None:
                st.metric(label="Audio Duration", value=f"{metrics['audio_duration']:.2f} s")
            else:
                st.metric(label="Audio Duration", value="N/A")
        
        # Performance metrics
        st.caption("Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Model", value=metrics["model"])
        with col2:
            st.metric(label="Total Duration", value=format_duration(metrics["total_duration_ns"]))
        with col3:
            if metrics["real_time_factor"] is not None:
                st.metric(label="Real-time Factor", value=f"{metrics['real_time_factor']:.2f}x")
            else:
                st.metric(label="Real-time Factor", value="N/A")
        
        # Output metrics
        st.caption("Output")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Word Count", value=f"{metrics['word_count']:,}")
        with col2:
            st.metric(label="Character Count", value=f"{metrics['character_count']:,}")
        with col3:
            st.metric(label="Segment Count", value=f"{metrics['segment_count']:,}")
    
    original_filename = st.session_state.filename.rsplit(".", 1)[0]
    download_filename = f"{original_filename}_transcript.json"
    
    # Prepare JSON for download
    export_data = {
        "source_file": st.session_state.filename,
        "transcript": st.session_state.transcript,
        "metrics": {
            "audio_file": {
                "file_size_bytes": metrics["file_size"],
                "audio_format": metrics["audio_format"],
                "audio_duration_seconds": metrics["audio_duration"],
            },
            "performance": {
                "model": metrics["model"],
                "total_duration_ns": metrics["total_duration_ns"],
                "real_time_factor": metrics["real_time_factor"],
            },
            "output": {
                "word_count": metrics["word_count"],
                "character_count": metrics["character_count"],
                "segment_count": metrics["segment_count"],
            },
        },
    }
    
    st.download_button(
        label="Download",
        data=json.dumps(export_data, indent=2),
        file_name=download_filename,
        mime="application/json"
    )
