import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    _display_transcription,
    _format_timestamp,
    _get_audio_duration,
    _handle_transcription,
    _show_detailed_analysis,
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
    assert ASR_MODEL_REPO == "mlx-community/whisper-turbo"


def test_audio_formats():
    assert "wav" in AUDIO_FORMATS
    assert "mp3" in AUDIO_FORMATS
    assert "m4a" in AUDIO_FORMATS


# --- _format_timestamp ---


def test_format_timestamp_seconds():
    assert _format_timestamp(3.2) == "00:03.2"


def test_format_timestamp_minutes():
    assert _format_timestamp(65.3) == "01:05.3"


def test_format_timestamp_zero():
    assert _format_timestamp(0.0) == "00:00.0"


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


@patch("streamlit_app.mlx_whisper")
def test_transcribe_success(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT

    result, elapsed = _transcribe(SAMPLE_AUDIO)

    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["avg_logprob"] == -0.25
    assert isinstance(elapsed, float)
    assert elapsed >= 0


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
        tab1, tab2 = MagicMock(), MagicMock()
        tab1.__enter__ = MagicMock(return_value=tab1)
        tab1.__exit__ = MagicMock(return_value=False)
        tab2.__enter__ = MagicMock(return_value=tab2)
        tab2.__exit__ = MagicMock(return_value=False)
        m.tabs.return_value = (tab1, tab2)
        m.columns.return_value = (MagicMock(), MagicMock())
        mock_event = MagicMock()
        mock_event.selection.rows = []
        m.dataframe.return_value = mock_event
        yield m


@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_stores_result(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    assert "transcription" in mock_st.session_state
    data = mock_st.session_state["transcription"]
    assert data["result"] == MOCK_WHISPER_RESULT
    assert data["eval_duration"] == 1.23
    assert data["audio_duration"] == 10.5
    assert data["file_stem"] == "interview_transcript"


@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=None)
def test_handle_transcription_stores_null_duration(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.warning.assert_called_once()
    assert mock_st.session_state["transcription"]["audio_duration"] is None


@patch(
    "streamlit_app._transcribe",
    side_effect=RuntimeError("Transcription produced no text"),
)
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_runtime_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once()
    assert "transcription" not in mock_st.session_state


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_unexpected_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Unexpected error: unexpected")
    mock_st.exception.assert_called_once()


@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_temp_dir_cleanup(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    called_paths = []
    mock_duration.side_effect = lambda p: (called_paths.append(p), 10.5)[1]

    _handle_transcription(mock_uploaded_file)

    assert len(called_paths) == 1
    assert called_paths[0].name == "audio.mp3"
    assert not called_paths[0].exists()


# --- _display_transcription ---


def test_display_transcription_no_session_state(mock_st):
    _display_transcription()

    mock_st.tabs.assert_not_called()


def test_display_transcription_shows_caption_and_tabs(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "eval_duration": 1.23,
        "audio_duration": 10.5,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.caption.assert_called_once_with("10.5s audio · 2 words · transcribed in 1.23s")
    mock_st.tabs.assert_called_once_with(["Transcript", "Detailed Analysis"])
    mock_st.code.assert_called_once_with("Hello world", language=None, wrap_lines=True)


def test_display_transcription_without_duration(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "eval_duration": 1.23,
        "audio_duration": None,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.caption.assert_called_once_with("2 words · transcribed in 1.23s")


# --- _show_detailed_analysis ---


def test_show_detailed_analysis_renders_segment_dataframe(mock_st):
    _show_detailed_analysis(MOCK_WHISPER_RESULT["segments"])

    assert mock_st.dataframe.called
    call_args = mock_st.dataframe.call_args_list[0]
    df = call_args[0][0]
    assert "#" in df.columns
    assert "Start" in df.columns
    assert "End" in df.columns
    assert "Text" in df.columns
    assert "Avg Log Prob" in df.columns
    assert "No Speech Prob" in df.columns
    assert "Compression Ratio" in df.columns
    assert "Temperature" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["#"] == 0
    assert df.iloc[0]["Avg Log Prob"] == -0.25


def test_show_detailed_analysis_no_segments(mock_st):
    _show_detailed_analysis([])

    mock_st.info.assert_called_once_with("No segment detail available.")
    mock_st.dataframe.assert_not_called()


def test_show_detailed_analysis_with_row_selected(mock_st):
    mock_event = MagicMock()
    mock_event.selection.rows = [0]
    mock_st.dataframe.return_value = mock_event

    _show_detailed_analysis(MOCK_WHISPER_RESULT["segments"])

    # Two dataframe calls: segment table + word table
    assert mock_st.dataframe.call_count == 2
    word_df = mock_st.dataframe.call_args_list[1][0][0]
    assert "Word" in word_df.columns
    assert "Probability" in word_df.columns
    assert len(word_df) == 2
    assert word_df.iloc[0]["Word"] == "Hello"
    assert word_df.iloc[0]["Probability"] == 0.98
