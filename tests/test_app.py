from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    _display_transcription,
    _handle_transcription,
    _transcribe,
)

SAMPLE_AUDIO = Path(__file__).parent / "data" / "audio" / "sample_10s.mp3"

MOCK_WHISPER_RESULT = {
    "text": "Hello world",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.5,
            "text": " Hello world",
            "tokens": [50364, 2425, 1002, 50414],
            "temperature": 0.0,
            "avg_logprob": -0.25,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "words": [
                {"word": " Hello", "start": 0.0, "end": 1.0, "probability": 0.98},
                {"word": " world", "start": 1.0, "end": 2.5, "probability": 0.95},
            ],
        }
    ],
    "language": "en",
}


# --- Constants ---


def test_asr_model_repo():
    assert ASR_MODEL_REPO == "mlx-community/whisper-large-v3-turbo"


def test_audio_formats():
    assert "wav" in AUDIO_FORMATS
    assert "mp3" in AUDIO_FORMATS
    assert "m4a" in AUDIO_FORMATS
    assert "mp4" in AUDIO_FORMATS
    assert "mov" in AUDIO_FORMATS


# --- _transcribe ---


@patch("streamlit_app.mlx_whisper")
def test_transcribe_success(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT

    result = _transcribe(SAMPLE_AUDIO)

    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["avg_logprob"] == -0.25


@patch("streamlit_app.mlx_whisper")
def test_transcribe_calls_mlx_with_correct_params(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT

    _transcribe(SAMPLE_AUDIO)

    mock_mlx.transcribe.assert_called_once_with(
        str(SAMPLE_AUDIO),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="en",
        task="transcribe",
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )


@patch("streamlit_app.mlx_whisper")
def test_transcribe_no_text_raises(mock_mlx):
    mock_mlx.transcribe.return_value = {"text": "   ", "segments": [], "language": "en"}

    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(SAMPLE_AUDIO)


# --- _handle_transcription ---


@pytest.fixture
def mock_uploaded_file():
    uploaded = MagicMock()
    uploaded.name = "interview.mp3"
    uploaded.read.return_value = b"fake audio bytes"
    return uploaded


@pytest.fixture
def mock_st():
    with patch("streamlit_app.st") as m:
        m.spinner.return_value.__enter__ = MagicMock()
        m.spinner.return_value.__exit__ = MagicMock(return_value=False)
        m.session_state = {}
        yield m


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_result(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(mock_uploaded_file)

    assert "transcription" in mock_st.session_state
    data = mock_st.session_state["transcription"]
    assert data["result"] == MOCK_WHISPER_RESULT
    assert data["file_stem"] == "interview_transcript"


@patch(
    "streamlit_app._transcribe",
    side_effect=RuntimeError("Transcription produced no text"),
)
def test_handle_transcription_runtime_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Transcription failed: Transcription produced no text")
    assert "transcription" not in mock_st.session_state


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
def test_handle_transcription_unexpected_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Unexpected error: unexpected")
    mock_st.exception.assert_called_once()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_temp_dir_cleanup(mock_transcribe, mock_st, mock_uploaded_file):
    called_paths = []
    mock_transcribe.side_effect = lambda p: (called_paths.append(p), MOCK_WHISPER_RESULT)[1]

    _handle_transcription(mock_uploaded_file)

    assert len(called_paths) == 1
    assert called_paths[0].name == "audio.mp3"
    assert not called_paths[0].exists()


# --- _display_transcription ---


def test_display_transcription_no_session_state(mock_st):
    _display_transcription()

    mock_st.text_area.assert_not_called()


def test_display_transcription_shows_transcript(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript", "Hello world", height=300, disabled=True, label_visibility="collapsed"
    )


def test_display_transcription_download_button(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download", "Hello world", "interview_transcript.txt", "text/plain"
    )
