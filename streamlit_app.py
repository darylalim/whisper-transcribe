import tempfile
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

# Model mapping for Whisper models
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

def asr_pipeline_conversion(audio_path: Path, model_name: str = "turbo") -> DoclingDocument:
    """Run the ASR pipeline and return a `DoclingDocument` transcript."""
    assert audio_path.exists(), f"Audio file not found: {audio_path}"

    converter = get_asr_converter(model_name)

    # Convert the audio file
    result: ConversionResult = converter.convert(audio_path)

    # Verify conversion was successful
    assert result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {result.status}"
    )
    return result.document

st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown text using the MLX Whisper models on Apple Silicon devices.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3"],
    accept_multiple_files=False,
    help="Accepts .wav and .mp3 files."
)

# Model selection
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
            # Generate transcript with spinner
            with st.spinner("Transcribing..."):
                doc = asr_pipeline_conversion(audio_path=tmp_path, model_name=selected_model)
                transcript = doc.export_to_markdown()
            
            # Store transcript in session state
            st.session_state.transcript = transcript
            st.session_state.filename = uploaded_file.name
            
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
    
    original_filename = st.session_state.filename.rsplit(".", 1)[0]
    download_filename = f"{original_filename}_transcript.md"
    
    st.download_button(
        label="Download",
        data=st.session_state.transcript,
        file_name=download_filename,
        mime="text/markdown"
    )
