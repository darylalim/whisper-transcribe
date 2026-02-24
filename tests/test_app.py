import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    ARTIFACTS_PATH,
    ASR_MODEL,
    AUDIO_FORMATS,
    _get_audio_duration,
    _handle_transcription,
    _transcribe,
)

SAMPLE_AUDIO = Path(__file__).parent / "data" / "audio" / "sample_10s.mp3"


@pytest.fixture
def mock_converter():
    with (
        patch("streamlit_app._get_converter") as mock_get_converter,
        patch("streamlit_app.ConversionStatus") as mock_status,
    ):
        mock_status.SUCCESS = "success"
        converter = MagicMock()
        mock_get_converter.return_value = converter
        yield converter


# --- Constants ---


def test_asr_model_is_turbo():
    from docling.datamodel import asr_model_specs

    assert ASR_MODEL is asr_model_specs.WHISPER_TURBO_MLX


def test_audio_formats():
    assert "wav" in AUDIO_FORMATS
    assert "mp3" in AUDIO_FORMATS
    assert "m4a" in AUDIO_FORMATS


def test_artifacts_path_is_absolute():
    assert Path(ARTIFACTS_PATH).is_absolute()


# --- _get_audio_duration ---


def test_get_audio_duration_with_sample_file():
    duration = _get_audio_duration(SAMPLE_AUDIO)
    assert duration is not None
    assert duration == pytest.approx(10.0, abs=0.5)


def test_get_audio_duration_nonexistent_file():
    result = _get_audio_duration(Path("/nonexistent/audio.mp3"))
    assert result is None


def test_get_audio_duration_invalid_file(tmp_path):
    bad_file = tmp_path / "not_audio.txt"
    bad_file.write_text("this is not audio")
    result = _get_audio_duration(bad_file)
    assert result is None


@patch(
    "streamlit_app.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
)
def test_get_audio_duration_timeout(mock_run):
    result = _get_audio_duration(SAMPLE_AUDIO)
    assert result is None


@patch("streamlit_app.subprocess.run")
def test_get_audio_duration_unparseable_output(mock_run):
    mock_run.return_value = MagicMock(stdout="not_a_number\n")
    result = _get_audio_duration(SAMPLE_AUDIO)
    assert result is None


@patch("streamlit_app.subprocess.run")
def test_get_audio_duration_parses_stdout(mock_run):
    mock_run.return_value = MagicMock(stdout="42.5\n")
    result = _get_audio_duration(SAMPLE_AUDIO)
    assert result == 42.5


# --- _transcribe ---


def test_transcribe_success(mock_converter):
    mock_result = MagicMock()
    mock_result.status = "success"
    mock_result.document.export_to_markdown.return_value = "Hello world"
    mock_converter.convert.return_value = mock_result

    transcript, elapsed = _transcribe(SAMPLE_AUDIO)

    assert transcript == "Hello world"
    assert isinstance(elapsed, float)
    assert elapsed >= 0


def test_transcribe_failure_raises_runtime_error(mock_converter):
    mock_result = MagicMock()
    mock_result.status = "failure"
    mock_converter.convert.return_value = mock_result

    with pytest.raises(RuntimeError, match="Conversion failed"):
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
        m.columns.return_value = (MagicMock(), MagicMock())
        yield m


@patch("streamlit_app._transcribe", return_value=("Hello world", 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_with_duration(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.caption.assert_called_once_with("10.5s audio · 2 words · transcribed in 1.23s")
    mock_st.code.assert_called_once_with("Hello world", language=None, wrap_lines=True)
    col1, col2 = mock_st.columns.return_value
    col1.download_button.assert_called_once()
    txt_args = col1.download_button.call_args
    assert txt_args[0][1] == "Hello world"
    assert txt_args[0][2] == "interview_transcript.txt"
    col2.download_button.assert_called_once()
    payload = json.loads(col2.download_button.call_args[0][1])
    assert payload == {
        "audio_duration": 10.5,
        "transcript": "Hello world",
        "num_words": 2,
        "eval_duration": 1.23,
    }


@patch("streamlit_app._transcribe", return_value=("Hello world", 1.23))
@patch("streamlit_app._get_audio_duration", return_value=None)
def test_handle_transcription_without_duration(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.warning.assert_called_once()
    mock_st.caption.assert_called_once_with("2 words · transcribed in 1.23s")
    payload = json.loads(mock_st.columns.return_value[1].download_button.call_args[0][1])
    assert payload["audio_duration"] is None


@patch("streamlit_app._transcribe", side_effect=RuntimeError("Conversion failed: failure"))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_runtime_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Transcription failed: Conversion failed: failure")


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_unexpected_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Unexpected error: unexpected")
    mock_st.exception.assert_called_once()


@patch("streamlit_app._transcribe", return_value=("Hello world", 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_temp_dir_cleanup(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    # Capture the path passed to _get_audio_duration to verify cleanup
    called_paths = []
    mock_duration.side_effect = lambda p: (called_paths.append(p), 10.5)[1]

    _handle_transcription(mock_uploaded_file)

    assert len(called_paths) == 1
    tmp_path = called_paths[0]
    assert tmp_path.name == "audio.mp3"
    assert not tmp_path.exists(), "Temp file should be cleaned up by TemporaryDirectory"
