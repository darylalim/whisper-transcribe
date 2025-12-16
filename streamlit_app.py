# Import packages
from pathlib import Path
import streamlit as st
import tempfile

from docling_core.types.doc import DoclingDocument

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

def get_asr_converter():
    """Create a DocumentConverter configured for ASR with automatic model selection.

    Uses `asr_model_specs.WHISPER_TURBO` which automatically selects the best
    implementation for your hardware:
    - MLX Whisper Turbo for Apple Silicon (M1/M2/M3) with mlx-whisper installed
    - Native Whisper Turbo as fallback

    You can swap in another model spec from `docling.datamodel.asr_model_specs`
    to experiment with different model sizes.
    """
    pipeline_options = AsrPipelineOptions(artifacts_path=artifacts_path)
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter

def asr_pipeline_conversion(audio_path: Path) -> DoclingDocument:
    """Run the ASR pipeline and return a `DoclingDocument` transcript."""
    # Check if the test audio file exists
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"

    converter = get_asr_converter()

    # Convert the audio file
    result: ConversionResult = converter.convert(audio_path)

    # Verify conversion was successful
    assert result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {result.status}"
    )
    return result.document

st.title("Automatic Speech Recognition Pipeline")
st.write("Transcribe audio files to Markdown text.")

# Drag and drop upload audio file
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["mp3"],
    accept_multiple_files=False,
    help="Accepts .mp3 only. Maximum file size is 2 MB."
)

# Check file size
MAX_FILE_SIZE = 2 * 1024 * 1024

if uploaded_file is not None:
    # Check file size
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE:
        st.error(f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds the 2 MB limit. Please upload a smaller file.")
    else:
        st.success(f"File uploaded: {uploaded_file.name} ({file_size / (1024 * 1024):.2f} MB)")
        
        # Button to generate transcript
        if st.button("Generate Transcript", type="primary"):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            try:
                # Generate transcript with spinner
                with st.spinner("Transcribing..."):
                    doc = asr_pipeline_conversion(audio_path=tmp_path)
                    transcript = doc.export_to_markdown()
                
                # Store transcript in session state
                st.session_state.transcript = transcript
                st.session_state.filename = uploaded_file.name
                
                st.success("Transcription complete!")
                
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")
            finally:
                # Clean up temporary file
                if tmp_path.exists():
                    tmp_path.unlink()

# Display transcript if available
if "transcript" in st.session_state:
    st.subheader("Transcript")
    st.markdown(st.session_state.transcript)
    
    # Button to download transcript
    original_filename = st.session_state.filename.rsplit(".", 1)[0]
    download_filename = f"{original_filename}_transcript.md"
    
    st.download_button(
        label="Download Transcript",
        data=st.session_state.transcript,
        file_name=download_filename,
        mime="text/markdown"
    )
