from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    _display_transcription,
    _fetch_url_audio,
    _fetch_youtube_audio,
    _format_srt,
    _format_timestamp,
    _handle_transcription,
    _RemoteAudio,
    _transcribe,
)

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
    assert AUDIO_FORMATS == (
        "mp3",
        "m4a",
        "wav",
        "flac",
        "ogg",
        "aac",
        "mp4",
        "mov",
        "webm",
        "mkv",
    )


# --- _RemoteAudio / _fetch_youtube_audio / _fetch_url_audio ---


def test_remote_audio_adapter():
    audio = _RemoteAudio("video.m4a", b"audio bytes")
    assert audio.name == "video.m4a"
    assert audio.read() == b"audio bytes"


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_returns_bytes_and_filename(mock_yt_dlp, tmp_path):
    fake_file = tmp_path / "Test_Video.m4a"
    fake_file.write_bytes(b"fake youtube audio")

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"title": "Test Video"}
    mock_ydl.prepare_filename.return_value = str(fake_file)
    mock_yt_dlp.YoutubeDL.return_value.__enter__.return_value = mock_ydl

    data, filename = _fetch_youtube_audio("https://youtube.com/watch?v=fetch_bytes")

    assert data == b"fake youtube audio"
    assert filename == "Test_Video.m4a"
    mock_ydl.extract_info.assert_called_once_with(
        "https://youtube.com/watch?v=fetch_bytes",
        download=True,
    )


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_uses_safe_options(mock_yt_dlp, tmp_path):
    fake_file = tmp_path / "video.webm"
    fake_file.write_bytes(b"webm bytes")

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"title": "video"}
    mock_ydl.prepare_filename.return_value = str(fake_file)
    mock_yt_dlp.YoutubeDL.return_value.__enter__.return_value = mock_ydl

    _fetch_youtube_audio("https://youtube.com/watch?v=safe_options")

    args, _ = mock_yt_dlp.YoutubeDL.call_args
    opts = args[0]
    assert opts["format"] == "bestaudio/best"
    assert opts["noplaylist"] is True
    assert opts["restrictfilenames"] is True
    assert opts["quiet"] is True


@patch("streamlit_app.urlopen")
def test_fetch_url_audio_returns_bytes_and_filename(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"file bytes"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    data, filename = _fetch_url_audio("https://example.com/audio.mp3")

    assert data == b"file bytes"
    assert filename == "audio.mp3"
    mock_urlopen.assert_called_once_with("https://example.com/audio.mp3", timeout=60)


@patch("streamlit_app.urlopen")
def test_fetch_url_audio_strips_query_from_filename(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"bytes"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    _, filename = _fetch_url_audio("https://example.com/path/audio.wav?t=42")

    assert filename == "audio.wav"


@patch("streamlit_app.urlopen")
def test_fetch_url_audio_decodes_percent_encoded_filename(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"bytes"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    _, filename = _fetch_url_audio("https://example.com/My%20Talk.mp3")

    assert filename == "My Talk.mp3"


@patch("streamlit_app.urlopen")
def test_fetch_url_audio_falls_back_when_no_path(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"bytes"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    _, filename = _fetch_url_audio("https://example.com/")

    assert filename == "download"


# --- _transcribe ---


@patch("streamlit_app.mlx_whisper")
def test_transcribe_success(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    result = _transcribe(b"fake audio", ".mp3")
    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1


@patch("streamlit_app.mlx_whisper")
def test_transcribe_calls_mlx_with_correct_params(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio params", ".mp3", "en", "transcribe")
    mock_mlx.transcribe.assert_called_once()
    args, kwargs = mock_mlx.transcribe.call_args
    assert args[0].endswith(".mp3")
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert kwargs["language"] == "en"
    assert kwargs["task"] == "transcribe"
    assert kwargs["initial_prompt"] is None
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["logprob_threshold"] == -1.0
    assert kwargs["compression_ratio_threshold"] == 2.4


@patch("streamlit_app.mlx_whisper")
def test_transcribe_defaults_language_to_none(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio default lang", ".mp3")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["language"] is None


@patch("streamlit_app.mlx_whisper")
def test_transcribe_defaults_task_to_transcribe(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio default task", ".mp3")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["task"] == "transcribe"


@patch("streamlit_app.mlx_whisper")
def test_transcribe_passes_translate_task(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio translate", ".mp3", "fr", "translate")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "fr"


@patch("streamlit_app.mlx_whisper")
def test_transcribe_defaults_initial_prompt_to_none(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio default prompt", ".mp3")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["initial_prompt"] is None


@patch("streamlit_app.mlx_whisper")
def test_transcribe_passes_initial_prompt(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio prompt", ".mp3", "en", "transcribe", "Anthropic, MLX")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["initial_prompt"] == "Anthropic, MLX"


@patch("streamlit_app.mlx_whisper")
def test_transcribe_defaults_no_verbatim_off(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio no verbatim default", ".mp3")
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["word_timestamps"] is False
    assert kwargs["hallucination_silence_threshold"] is None


@patch("streamlit_app.mlx_whisper")
def test_transcribe_no_verbatim_enables_hallucination_skip(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT
    _transcribe(b"fake audio no verbatim on", ".mp3", None, "transcribe", None, True)
    _, kwargs = mock_mlx.transcribe.call_args
    assert kwargs["word_timestamps"] is True
    assert kwargs["hallucination_silence_threshold"] == 2.0


@patch("streamlit_app.mlx_whisper")
def test_transcribe_no_text_raises(mock_mlx):
    mock_mlx.transcribe.return_value = {"text": "   ", "segments": [], "language": "en"}
    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(b"fake audio empty", ".mp3")


@patch("streamlit_app.mlx_whisper")
def test_transcribe_cleans_up_temp_file(mock_mlx):
    called_paths = []

    def capture_path(path, **kwargs):
        called_paths.append(path)
        return MOCK_WHISPER_RESULT

    mock_mlx.transcribe.side_effect = capture_path
    _transcribe(b"fake audio cleanup", ".mp3")
    assert len(called_paths) == 1
    assert not Path(called_paths[0]).exists()


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
        m.session_state = {}
        m.columns.side_effect = lambda spec, **_: [
            MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        m.text_area.side_effect = lambda label, value, **_: value
        yield m


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_result(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], None, "transcribe", False)

    assert "transcription" in mock_st.session_state
    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 1
    data = transcriptions[0]
    assert data["result"] == MOCK_WHISPER_RESULT
    assert data["file_stem"] == "interview_mp3_transcript"
    assert data["filename"] == "interview.mp3"
    assert data["include_subtitles"] is False


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_include_subtitles_true(
    mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription([mock_uploaded_file], None, "transcribe", True)
    data = mock_st.session_state["transcription"][0]
    assert data["include_subtitles"] is True


@patch(
    "streamlit_app._transcribe",
    side_effect=RuntimeError("Transcription produced no text"),
)
def test_handle_transcription_runtime_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], None, "transcribe", False)

    mock_st.error.assert_called_once_with(
        "Transcription failed for interview.mp3: Transcription produced no text"
    )
    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
def test_handle_transcription_unexpected_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], None, "transcribe", False)

    mock_st.error.assert_called_once_with("Unexpected error for interview.mp3: unexpected")
    mock_st.exception.assert_called_once()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_passes_args(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], "fr", "translate", True)
    mock_transcribe.assert_called_once_with(
        b"fake audio bytes", ".mp3", "fr", "translate", None, False
    )


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_passes_initial_prompt(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], None, "transcribe", False, "Anthropic, MLX")
    mock_transcribe.assert_called_once_with(
        b"fake audio bytes", ".mp3", None, "transcribe", "Anthropic, MLX", False
    )


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_passes_no_verbatim(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription([mock_uploaded_file], None, "transcribe", False, None, True)
    mock_transcribe.assert_called_once_with(
        b"fake audio bytes", ".mp3", None, "transcribe", None, True
    )


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_multiple_files(mock_transcribe, mock_st):
    file1 = MagicMock()
    file1.name = "first.mp3"
    file1.read.return_value = b"first audio"
    file2 = MagicMock()
    file2.name = "second.mp3"
    file2.read.return_value = b"second audio"

    _handle_transcription([file1, file2], None, "transcribe", False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 2
    assert transcriptions[0]["filename"] == "first.mp3"
    assert transcriptions[1]["filename"] == "second.mp3"
    assert mock_transcribe.call_count == 2


@patch("streamlit_app._transcribe")
def test_handle_transcription_partial_failure(mock_transcribe, mock_st):
    mock_transcribe.side_effect = [
        MOCK_WHISPER_RESULT,
        RuntimeError("Transcription produced no text"),
        MOCK_WHISPER_RESULT,
    ]
    files = []
    for stem in ("first", "second", "third"):
        f = MagicMock()
        f.name = f"{stem}.mp3"
        f.read.return_value = b"bytes"
        files.append(f)

    _handle_transcription(files, None, "transcribe", False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 2
    assert transcriptions[0]["filename"] == "first.mp3"
    assert transcriptions[1]["filename"] == "third.mp3"
    mock_st.error.assert_called_once_with(
        "Transcription failed for second.mp3: Transcription produced no text"
    )


# --- _display_transcription ---


def test_display_transcription_no_session_state(mock_st):
    _display_transcription()

    mock_st.text_area.assert_not_called()


def test_display_transcription_shows_transcript(mock_st):
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "interview_transcript",
            "filename": "interview.mp3",
            "include_subtitles": False,
        }
    ]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        "Hello world",
        height=300,
        label_visibility="collapsed",
        key="transcript_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_txt_download(mock_st):
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "interview_transcript",
            "filename": "interview.mp3",
            "include_subtitles": False,
        }
    ]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        ".txt",
        "Hello world",
        "interview_transcript.txt",
        "text/plain",
        key="download_txt_0",
        use_container_width=True,
    )


def test_display_transcription_srt_download(mock_st):
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "interview_transcript",
            "filename": "interview.mp3",
            "include_subtitles": True,
        }
    ]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        ".srt",
        "1\n00:00:00,000 --> 00:00:02,500\nHello world\n",
        "interview_transcript.srt",
        "application/x-subrip",
        key="download_srt_0",
        use_container_width=True,
    )


def test_display_transcription_subtitles_on(mock_st):
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "interview_transcript",
            "filename": "interview.mp3",
            "include_subtitles": True,
        }
    ]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        "1\n00:00:00,000 --> 00:00:02,500\nHello world\n",
        height=300,
        label_visibility="collapsed",
        key="transcript_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_download_reflects_edits(mock_st):
    mock_st.text_area.side_effect = lambda label, value, **_: "edited transcript text"
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "interview_transcript",
            "filename": "interview.mp3",
            "include_subtitles": False,
        }
    ]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        ".txt",
        "edited transcript text",
        "interview_transcript.txt",
        "text/plain",
        key="download_txt_0",
        use_container_width=True,
    )


def test_display_transcription_multiple_files(mock_st):
    mock_st.session_state["transcription"] = [
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "first_transcript",
            "filename": "first.mp3",
            "include_subtitles": False,
        },
        {
            "result": MOCK_WHISPER_RESULT,
            "file_stem": "second_transcript",
            "filename": "second.mp3",
            "include_subtitles": False,
        },
    ]

    _display_transcription()

    assert mock_st.text_area.call_count == 2
    assert mock_st.download_button.call_count == 2
    assert mock_st.subheader.call_count == 2
    mock_st.subheader.assert_any_call("first.mp3")
    mock_st.subheader.assert_any_call("second.mp3")


# --- formatting helpers ---


def test_format_timestamp_zero():
    assert _format_timestamp(0.0) == "00:00:00.000"


def test_format_timestamp_minutes_and_seconds():
    assert _format_timestamp(65.5) == "00:01:05.500"


def test_format_timestamp_hours():
    assert _format_timestamp(3661.123) == "01:01:01.123"


def test_format_timestamp_with_comma_decimal_marker():
    assert _format_timestamp(65.5, decimal_marker=",") == "00:01:05,500"


def test_format_srt():
    assert _format_srt(MOCK_WHISPER_RESULT) == ("1\n00:00:00,000 --> 00:00:02,500\nHello world\n")


def test_format_srt_multiple_segments():
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello"},
            {"start": 2.5, "end": 5.0, "text": " World"},
        ]
    }
    assert _format_srt(result) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello\n\n2\n00:00:02,500 --> 00:00:05,000\nWorld\n"
    )


def test_format_srt_escapes_arrow():
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " before --> after"},
        ]
    }
    assert _format_srt(result) == "1\n00:00:00,000 --> 00:00:02,500\nbefore -> after\n"
