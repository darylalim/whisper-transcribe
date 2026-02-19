import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    ARTIFACTS_PATH,
    AUDIO_FORMATS,
    MODEL_NAMES,
    MODEL_OPTIONS,
    _get_audio_duration,
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


def test_model_options_has_six_models():
    assert len(MODEL_OPTIONS) == 6


def test_model_names_matches_model_options_keys():
    assert MODEL_NAMES == tuple(MODEL_OPTIONS)


def test_model_names_is_tuple():
    assert isinstance(MODEL_NAMES, tuple)


def test_turbo_in_model_names():
    assert "turbo" in MODEL_NAMES


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

    transcript, elapsed = _transcribe(SAMPLE_AUDIO, "tiny")

    assert transcript == "Hello world"
    assert isinstance(elapsed, float)
    assert elapsed >= 0


def test_transcribe_failure_raises_runtime_error(mock_converter):
    mock_result = MagicMock()
    mock_result.status = "failure"
    mock_converter.convert.return_value = mock_result

    with pytest.raises(RuntimeError, match="Conversion failed"):
        _transcribe(SAMPLE_AUDIO, "tiny")
