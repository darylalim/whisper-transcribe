from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import streamlit as st
from streamlit.elements.lib.file_uploader_utils import normalize_upload_file_type
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.memory_media_file_storage import get_extension_for_mimetype
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    DEFAULT_MEDIA_MIME,
    ERROR_ICON,
    ERROR_MESSAGE_LIMIT,
    FORMAT_PLAIN_TEXT,
    FORMAT_SUBTITLES,
    MEDIA_MIME_TYPES,
    PAGE_CONFIG,
    SELECT_WIDTH,
    SUBTITLE_LINE_WIDTH,
    TRANSCRIPT_FORMATS,
    VIDEO_FORMATS,
    _active_sources,
    _condense,
    _display_transcription,
    _error,
    _escape_markdown,
    _format_language,
    _format_srt,
    _format_timestamp,
    _handle_transcription,
    _media_mime,
    _plural,
    _split_clips,
    _transcribe,
    _transcription_kwargs,
    _validate_time_range,
    _wrap_cue,
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

SRT_HELLO = "1\n00:00:00,000 --> 00:00:02,500\nHello world\n"

# Spelled out rather than imported from streamlit_app — this *is* the assertion,
# so reading the app's own value would make the pin tautological. The second
# sentence is load-bearing: st.download_button bakes its payload at render time,
# so an uncommitted text-area edit is silently dropped (verified in Chrome).
DOWNLOAD_HELP = (
    "Downloads as .srt when subtitles are enabled, .txt otherwise. "
    "Commit an edit first — click outside the box or press "
    "Ctrl/Cmd+Enter — or the download will miss it."
)


# --- Helpers ---


def _make_file(name="interview.mp3", data=b"fake audio bytes"):
    f = MagicMock()
    f.name = name
    f.read.return_value = data
    return f


def _make_transcription(
    include_subtitles=False,
    file_stem="interview_transcript",
    filename="interview.mp3",
    text=None,
):
    result = MOCK_WHISPER_RESULT
    if text is not None:
        # Only the keys _display_transcription and _format_srt read.
        result = {"text": text, "segments": [{"start": 0.0, "end": 2.5, "text": text}]}
    return {
        "result": result,
        "file_stem": file_stem,
        "filename": filename,
        "include_subtitles": include_subtitles,
    }


def _expected_transcribe_kwargs(**overrides):
    base = {
        "language": None,
        "task": "transcribe",
        "initial_prompt": None,
        "no_verbatim": False,
        "condition_on_previous_text": True,
        "clip_timestamps": "0",
    }
    return base | overrides


def _handle_transcription_kwargs(**overrides):
    return {"language": None, "task": "transcribe", "include_subtitles": False} | overrides


def _ui_state(**overrides):
    base = {
        "language": None,
        "translate": False,
        "include_subtitles": False,
        "initial_prompt": None,
        "no_verbatim": False,
        "decode_independently": False,
        "clip_timestamps": "0",
    }
    return base | overrides


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_caches():
    _transcribe.clear()
    # The wrapper above only reaches the cache created by the imported module.
    # AppTest re-executes the script as a separate module with its own cache
    # store, which would otherwise persist for the whole session and let a
    # later case read a stale hit from an earlier one.
    st.cache_data.clear()


@pytest.fixture
def mock_mlx():
    with patch("streamlit_app.mlx_whisper") as m:
        m.transcribe.return_value = MOCK_WHISPER_RESULT
        yield m


@pytest.fixture
def mock_uploaded_file():
    return _make_file()


@pytest.fixture
def mock_st():
    with patch("streamlit_app.st") as m:
        m.session_state = {}
        m.columns.side_effect = lambda spec, **_: [
            MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        m.text_area.side_effect = lambda label, value, **_: value
        yield m


# --- Constants ---


def test_asr_model_repo():
    assert ASR_MODEL_REPO == "mlx-community/whisper-large-v3-turbo"


def test_transcript_formats():
    # Order is display order in the segmented control, and index 0 is the default.
    assert TRANSCRIPT_FORMATS == (FORMAT_PLAIN_TEXT, FORMAT_SUBTITLES)
    assert TRANSCRIPT_FORMATS == ("Plain text", "Subtitles")


def test_audio_formats():
    assert AUDIO_FORMATS == (
        "mp3",
        "m4a",
        "wav",
        "opus",
    )


def test_video_formats():
    assert VIDEO_FORMATS == (
        "mp4",
        "mov",
        "webm",
        "mkv",
    )


def test_format_list_fits_the_dropzone_hint():
    # The dropzone renders "<size> per file • MP3, M4A, ..." on a single
    # `white-space: nowrap; text-overflow: ellipsis` line with ~479px for the
    # format list at the centered layout's max width. Measured against Source
    # Sans 14px, each entry costs ~31-40px, so the list has to stay short.
    hint = ", ".join(f.upper() for f in AUDIO_FORMATS + VIDEO_FORMATS)
    assert len(hint) <= 60, f"{hint!r} will truncate in the uploader dropzone"


# --- _transcribe ---


def test_transcribe_success(mock_mlx):
    result = _transcribe(b"fake audio", ".mp3")
    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1


def test_transcribe_calls_mlx_with_correct_params(mock_mlx):
    _transcribe(b"fake audio params", ".mp3", language="en", task="transcribe")
    call = mock_mlx.transcribe.call_args
    assert call.args[0].endswith(".mp3")
    assert call.kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert call.kwargs["language"] == "en"
    assert call.kwargs["task"] == "transcribe"
    assert call.kwargs["no_speech_threshold"] == 0.6
    assert call.kwargs["logprob_threshold"] == -1.0
    assert call.kwargs["compression_ratio_threshold"] == 2.4


def test_transcribe_defaults(mock_mlx):
    _transcribe(b"fake audio defaults", ".mp3")
    kwargs = mock_mlx.transcribe.call_args.kwargs
    assert kwargs["language"] is None
    assert kwargs["task"] == "transcribe"
    assert kwargs["initial_prompt"] is None
    assert kwargs["word_timestamps"] is False
    assert kwargs["hallucination_silence_threshold"] is None
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["clip_timestamps"] == "0"


@pytest.mark.parametrize(
    "call_kwargs,expected",
    [
        ({"language": "fr", "task": "translate"}, {"task": "translate", "language": "fr"}),
        ({"initial_prompt": "Anthropic, MLX"}, {"initial_prompt": "Anthropic, MLX"}),
        ({"no_verbatim": True}, {"word_timestamps": True, "hallucination_silence_threshold": 2.0}),
        ({"condition_on_previous_text": False}, {"condition_on_previous_text": False}),
        ({"clip_timestamps": "30,90"}, {"clip_timestamps": "30,90"}),
    ],
    ids=["translate", "initial_prompt", "no_verbatim", "no_context", "single_clip"],
)
def test_transcribe_forwards_kwargs(mock_mlx, call_kwargs, expected):
    _transcribe(b"audio", ".mp3", **call_kwargs)
    kwargs = mock_mlx.transcribe.call_args.kwargs
    assert {k: kwargs[k] for k in expected} == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", ["0"]),
        ("30,90", ["30,90"]),
        ("0,60,120,180", ["0,60", "120,180"]),
        ("0,60,120", ["0,60", "120"]),
        (" 30 , 90 ", ["30,90"]),
        ("", ["0"]),
    ],
    ids=["full_file", "single_pair", "two_pairs", "trailing_start", "whitespace", "blank"],
)
def test_split_clips(raw, expected):
    assert _split_clips(raw) == expected


def test_transcribe_splits_multi_clip_ranges(mock_mlx):
    # mlx_whisper does not honour more than one clip: its decode loop binds
    # seek_clip_start and never re-seeks to it, so "0,60,120,180" decoded 0-180
    # straight through -- including the 60-120 the user excluded. One call per
    # pair is the workaround. This case replaces a `multi_clip` parametrize entry
    # that asserted only that the string was *forwarded*, which is exactly why a
    # fully green suite shipped the wrong audio for the app's own README example.
    _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")

    calls = [c.kwargs["clip_timestamps"] for c in mock_mlx.transcribe.call_args_list]
    assert calls == ["0,60", "120,180"]


def test_transcribe_keeps_a_trailing_start_open_ended(mock_mlx):
    # An odd trailing value is a start that runs to the end of the file -- that is
    # mlx_whisper's own reading of clip_timestamps -- so it must stay unpaired
    # rather than being padded into a degenerate range.
    _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120")

    calls = [c.kwargs["clip_timestamps"] for c in mock_mlx.transcribe.call_args_list]
    assert calls == ["0,60", "120"]


def test_transcribe_single_clip_makes_one_call(mock_mlx):
    # The common path stays exactly one model pass with the string unchanged.
    _transcribe(b"audio", ".mp3", clip_timestamps="30,90")

    assert mock_mlx.transcribe.call_count == 1
    assert mock_mlx.transcribe.call_args.kwargs["clip_timestamps"] == "30,90"


def test_transcribe_merges_multi_clip_results(mock_mlx):
    # Segment timestamps are already absolute (mlx derives them from `seek`, which
    # starts at the clip's own start frame), so segments concatenate with no
    # timestamp fix-up. Both clips carry id 0 because mlx numbers per call
    # (`enumerate(current_segments, start=len(all_segments))`), so a plain
    # concatenation would hand back duplicates -- a shape no single call produces.
    mock_mlx.transcribe.side_effect = [
        {
            "text": " one",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": " one"}],
            "language": "en",
        },
        {
            "text": " two",
            "segments": [{"id": 0, "start": 120.0, "end": 121.0, "text": " two"}],
            "language": "fr",
        },
    ]

    result = _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")

    assert result["text"] == " one two"
    assert [s["start"] for s in result["segments"]] == [0.0, 120.0]
    assert [s["id"] for s in result["segments"]] == [0, 1]
    # Not "fr" -- though the clips cannot actually disagree: with language=None
    # mlx detects from the first 30s of the *whole file*, not of the clip, so
    # every call scores the same window. The mock differs only to pin which wins.
    assert result["language"] == "en"


def test_transcribe_raises_only_when_every_clip_is_silent(mock_mlx):
    # A silent clip inside a multi-clip range must not fail the whole file: the
    # empty-text guard applies to the merged text, not to each pass.
    silent = {"text": "   ", "segments": [], "language": "en"}
    spoken = {
        "text": " hello",
        "segments": [{"start": 120.0, "end": 121.0, "text": " hello"}],
        "language": "en",
    }
    mock_mlx.transcribe.side_effect = [silent, spoken]
    assert _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")["text"].strip() == "hello"

    mock_mlx.transcribe.side_effect = [silent, silent]
    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(b"other audio", ".mp3", clip_timestamps="0,60,120,180")


def test_transcribe_no_text_raises(mock_mlx):
    mock_mlx.transcribe.return_value = {"text": "   ", "segments": [], "language": "en"}
    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(b"fake audio empty", ".mp3")


def test_transcribe_cleans_up_temp_file(mock_mlx):
    called_paths = []

    def capture(path, **_):
        called_paths.append(path)
        return MOCK_WHISPER_RESULT

    mock_mlx.transcribe.side_effect = capture
    _transcribe(b"fake audio cleanup", ".mp3")
    assert len(called_paths) == 1
    assert not Path(called_paths[0]).exists()


# --- _handle_transcription ---


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_result(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )

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
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=True
    )
    assert mock_st.session_state["transcription"][0]["include_subtitles"] is True


@patch("streamlit_app._transcribe", side_effect=RuntimeError("Transcription produced no text"))
def test_handle_transcription_runtime_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )
    # `\:` renders as a plain colon. _error escapes the whole message rather than
    # only the untrusted spans, so the fixed literals pick up backslashes too.
    mock_st.error.assert_called_once_with(
        r"Transcription failed for interview.mp3\: Transcription produced no text",
        icon=":material/error:",
    )
    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
def test_handle_transcription_unexpected_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )
    mock_st.error.assert_called_once_with(
        r"Unexpected error for interview.mp3\: unexpected", icon=":material/error:"
    )
    mock_st.exception.assert_called_once()


@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("boom"), r"Transcription failed for my\_song \[live\].mp3\: boom"),
        (ValueError("boom"), r"Unexpected error for my\_song \[live\].mp3\: boom"),
    ],
)
def test_handle_transcription_escapes_filename_in_error(error, expected, mock_st):
    # st.error renders *full* Markdown — a strictly larger subset than the label
    # subset _display_transcription's subheader escapes for. Filenames are
    # whatever the browser sent, so they are untrusted.
    with patch("streamlit_app._transcribe", side_effect=error):
        _handle_transcription(
            [_make_file(name="my_song [live].mp3")], **_handle_transcription_kwargs()
        )

    mock_st.error.assert_called_once_with(expected, icon=":material/error:")


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_escapes_filename_in_status_label(mock_transcribe, mock_st):
    # The status label renders on every file, not just failures, and its Markdown
    # subset includes images — so an unescaped `![](https://host/x.png)` in a
    # filename would fetch on the happy path.
    _handle_transcription([_make_file(name="clip [1].mp3")], **_handle_transcription_kwargs())

    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_any_call(label=r"Transcribing clip \[1\].mp3 (1/1)...")


@pytest.mark.parametrize(
    "count,opening,closing",
    [
        (1, "Transcribing 1 file...", "Transcribed 1/1 file"),
        (2, "Transcribing 2 files...", "Transcribed 2/2 files"),
    ],
    ids=["single", "batch"],
)
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_pluralizes_the_status_labels(
    mock_transcribe, mock_st, count, opening, closing
):
    # Both of these labels used to read "file(s)". Neither was asserted anywhere,
    # which is why the parenthesised form survived -- and a single file, the case
    # it gets wrong, is the common one.
    _handle_transcription(
        [_make_file(name=f"clip{i}.mp3") for i in range(count)],
        **_handle_transcription_kwargs(),
    )

    mock_st.status.assert_called_once_with(opening, expanded=True)
    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_called_with(label=closing, state="complete")


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Short and fine.", "Short and fine."),
        ("collapse   these\n\nspaces", "collapse these spaces"),
    ],
    ids=["passthrough", "whitespace"],
)
def test_condense_leaves_human_sized_messages_alone(message, expected):
    assert _condense(message) == expected


def test_condense_keeps_both_ends():
    # Head *and* tail, not truncation: the two useful halves of a long exception
    # sit at opposite ends. mlx_whisper raises RuntimeError("Failed to load
    # audio: " + ffmpeg's whole stderr) -- 1740 characters on a corrupt file, of
    # which ~1500 are the version banner and --enable-* flags, with the actual
    # diagnosis last. Truncating from the front would render only the banner.
    long_message = (
        "Transcription failed for clip.wav: Failed to load audio: "
        + "ffmpeg version 7.1.1 --enable-everything " * 40
        + "Invalid data found when processing input"
    )
    condensed = _condense(long_message)

    assert len(condensed) <= ERROR_MESSAGE_LIMIT + len(" […] ")
    assert condensed.startswith("Transcription failed for clip.wav:")
    assert condensed.endswith("Invalid data found when processing input")
    assert " […] " in condensed


def test_error_condenses_before_rendering(mock_st):
    # The cap has to be applied inside _error rather than at the call sites, for
    # the same reason the icon is: every caller, present or future, must be
    # unable to render an unbounded alert.
    _error("Unexpected error for clip.wav: " + "verbose library noise " * 200)

    (rendered,), _ = mock_st.error.call_args
    assert len(rendered) < 400
    assert "…" in rendered


def test_error_still_escapes_after_condensing(mock_st):
    # Condensing runs first and escaping second, so a payload surviving the cap
    # is still neutralised. The order matters: escaping first would let the slice
    # land between a backslash and the character it escapes, silently un-escaping
    # whatever sits on the cut.
    _error("Transcription failed for ![](p.png).mp3: Failed to load audio:" + " padding" * 200)

    (rendered,), _ = mock_st.error.call_args
    assert "![](" not in rendered


def test_error_condenses_before_escaping(mock_st):
    # The order is load-bearing, not incidental. Escaping first would let the cut
    # land *between* a backslash and the character it escapes, dropping the
    # backslash into the head and releasing a live `[` or `:` into the tail.
    #
    # The signature is the ellipsis marker itself: inserted before escaping it
    # comes out escaped, and inserted after it arrives raw. No boundary
    # arithmetic needed -- a first draft of this test computed the cut's parity to
    # land a backslash exactly on it, which was both fragile and unnecessary.
    _error("Failed: " + ":" * ERROR_MESSAGE_LIMIT * 4)

    (rendered,), _ = mock_st.error.call_args
    assert r"\[…\]" in rendered
    # And nothing lost its backslash on the way through the cut.
    assert ":" not in rendered.replace(r"\:", "")


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Transcription failed for ![](p.png).mp3: boom",
            # `(` and `)` are deliberately not in the class and do not need to be:
            # an image needs the bracket half, and `[`/`]` are escaped, so the
            # parenthesized destination is inert on its own.
            r"Transcription failed for !\[\](p.png).mp3\: boom",
        ),
        (
            # A filename cannot carry `/`, and does not need one to point off-site:
            # `https:host` is a valid image destination that the URL parser
            # resolves to https://host/ (verified with WHATWG `new URL`).
            "Transcription failed for ![](https:evil.example.com).mp3: boom",
            r"Transcription failed for !\[\](https\:evil.example.com).mp3\: boom",
        ),
        ("Invalid time range: '![](https://x/p.png)' is not a number.", None),
    ],
    ids=["filename_in_failure", "scheme_only_destination", "validation_message"],
)
def test_error_escapes_the_whole_message(mock_st, message, expected):
    # Two untrusted inputs reach the alert body verbatim: the upload's filename,
    # which _handle_transcription interpolates into both failure messages, and the
    # raw Time range text, which _validate_time_range echoes back. st.error's body
    # renders full Markdown, so an unescaped `![](...)` fires an image request to
    # whatever destination the text spells.
    _error(message)

    (rendered,), kwargs = mock_st.error.call_args
    assert kwargs == {"icon": ERROR_ICON}
    assert "![](" not in rendered
    if expected is not None:
        assert rendered == expected


def test_handle_transcription_rewinds_the_cursor_before_reading(mock_st):
    # UploadedFile subclasses io.BytesIO, and read() leaves it at EOF. The widget
    # hands the script a deepcopy of the cached value on every run (register_widget
    # in session_state.py), so no rerun delivers an exhausted object; what the
    # seek(0) guards is a same-object double read within one run, which is what
    # this test constructs. st.audio rewinds as a side effect of rendering a preview
    # (_marshall_av_media calls data.seek(0)), but the Record tab renders none.
    rec = UploadedFileRec("id", "recording.wav", "audio/wav", b"real audio bytes")
    recording = UploadedFile(rec, FileURLs())

    with patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT) as mock_transcribe:
        _handle_transcription([recording], **_handle_transcription_kwargs())
        _handle_transcription([recording], **_handle_transcription_kwargs())

    assert [c.args[0] for c in mock_transcribe.call_args_list] == [b"real audio bytes"] * 2


def test_handle_transcription_collapses_whitespace_in_filename(mock_st):
    # st.error's body is one of two sinks rendered without the frontend's isLabel
    # flag, which is what auto-escapes `#`/`>` and strips block elements
    # elsewhere — so an uncollapsed newline would open a heading and a blockquote
    # inside the error box. Every block construct needs a line start. No browser
    # path delivers a newline (browsers percent-encode one in an upload's name
    # before sending it), but a direct PUT to the upload route can, since the
    # route copies the multipart filename through unchanged.
    with patch("streamlit_app._transcribe", side_effect=RuntimeError("boom")):
        _handle_transcription(
            [_make_file(name="clip\n\n# Big\n\n> quote.mp3")], **_handle_transcription_kwargs()
        )

    # `#` and `>` are deliberately *not* in the escape class — losing the line
    # start is what defuses them, so they survive as literal characters.
    (message,), kwargs = mock_st.error.call_args
    assert "\n" not in message
    assert message == r"Transcription failed for clip # Big > quote.mp3\: boom"
    assert kwargs == {"icon": ":material/error:"}


@pytest.mark.parametrize(
    "ui_kwargs,expected_overrides",
    [
        (
            _handle_transcription_kwargs(language="fr", task="translate", include_subtitles=True),
            {"language": "fr", "task": "translate"},
        ),
        (
            _handle_transcription_kwargs(initial_prompt="Anthropic, MLX"),
            {"initial_prompt": "Anthropic, MLX"},
        ),
        (_handle_transcription_kwargs(no_verbatim=True), {"no_verbatim": True}),
        (
            _handle_transcription_kwargs(condition_on_previous_text=False),
            {"condition_on_previous_text": False},
        ),
        (_handle_transcription_kwargs(clip_timestamps="30,90"), {"clip_timestamps": "30,90"}),
    ],
    ids=["translate", "initial_prompt", "no_verbatim", "no_context", "clip"],
)
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_forwards_kwargs(
    mock_transcribe, mock_st, mock_uploaded_file, ui_kwargs, expected_overrides
):
    _handle_transcription([mock_uploaded_file], **ui_kwargs)
    mock_transcribe.assert_called_once_with(
        b"fake audio bytes",
        ".mp3",
        **_expected_transcribe_kwargs(**expected_overrides),
    )


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_multiple_files(mock_transcribe, mock_st):
    files = [_make_file("first.mp3", b"first audio"), _make_file("second.mp3", b"second audio")]
    _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

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
    files = [_make_file(f"{stem}.mp3") for stem in ("first", "second", "third")]

    _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 2
    assert transcriptions[0]["filename"] == "first.mp3"
    assert transcriptions[1]["filename"] == "third.mp3"
    mock_st.error.assert_called_once_with(
        r"Transcription failed for second.mp3\: Transcription produced no text",
        icon=":material/error:",
    )


@patch("streamlit_app._transcribe", side_effect=RuntimeError("boom"))
def test_handle_transcription_renders_errors_after_the_status_closes(mock_transcribe, mock_st):
    # st.status collapses on its first update(label=...) -- update() clears the
    # proto's `expanded` field unless it is passed again, and the frontend's
    # label-change branch then resets the open state to that now-false value. So
    # an alert written into the status body during the loop lands in a box the
    # user cannot open: a failed file rendered a green check, "Transcribed 0/1
    # file", and no visible explanation anywhere on the page. The fix is
    # positional, so the assertion is too -- st.error must run after the status
    # context manager has exited.
    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    names = [c[0] for c in mock_st.mock_calls]
    assert names.index("error") > names.index("status().__exit__")
    # And a batch that lost a file is not "complete".
    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_called_with(label="Transcribed 0/1 file", state="error")


@patch("streamlit_app._transcribe")
def test_handle_transcription_keeps_finished_files_when_interrupted(mock_transcribe, mock_st):
    # Streamlit aborts a running script by raising RerunException (a BaseException,
    # so `except Exception` does not catch it) at the next ForwardMsg — which
    # status.update() is. Results published so far must survive that unwind.
    class _Interrupt(BaseException):
        pass

    mock_transcribe.side_effect = [MOCK_WHISPER_RESULT, _Interrupt()]
    files = [_make_file(f"{stem}.mp3") for stem in ("first", "second")]

    with pytest.raises(_Interrupt):
        _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 1
    assert transcriptions[0]["filename"] == "first.mp3"


@patch("streamlit_app.mx")
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_reclaims_the_mlx_cache(mock_transcribe, mock_mx, mock_st):
    # mlx keeps freed device buffers on a free list instead of returning them, so
    # a finished batch stays resident for the life of the server process.
    # Measured against whisper-large-v3-turbo with word_timestamps=True: 894 MB
    # held after an 8-second file, 1.25 GB after two minutes, all reclaimed in
    # 4.2 ms, with the model itself untouched in `active` memory.
    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_mx.clear_cache.assert_called_once_with()


@patch("streamlit_app.mx")
def test_handle_transcription_reclaims_the_mlx_cache_when_interrupted(mock_mx, mock_st):
    # This is why it is a `finally` and not a call after the block. A tab switch
    # -- or any widget -- raises RerunException, a BaseException, mid-batch; an
    # interrupted run would otherwise hold its buffers until whenever the next
    # transcription happens to finish, which may be never.
    class _Interrupt(BaseException):
        pass

    with (
        patch("streamlit_app._transcribe", side_effect=_Interrupt()),
        pytest.raises(_Interrupt),
    ):
        _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_mx.clear_cache.assert_called_once_with()


@patch("streamlit_app._transcribe")
def test_handle_transcription_clears_previous_batch(mock_transcribe, mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="stale.mp3")]
    mock_transcribe.side_effect = RuntimeError("Transcription produced no text")

    _handle_transcription([_make_file()], language=None, task="transcribe", include_subtitles=False)

    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe")
def test_handle_transcription_bumps_batch_id(mock_transcribe, mock_st):
    # The batch id namespaces _display_transcription's widget keys, so every
    # batch must advance it — including one where every file fails, since the
    # stale text area would otherwise outlive the results it was rendered from.
    mock_transcribe.return_value = MOCK_WHISPER_RESULT
    kwargs = _handle_transcription_kwargs()

    _handle_transcription([_make_file()], **kwargs)
    assert mock_st.session_state["batch_id"] == 1

    _handle_transcription([_make_file("second.mp3")], **kwargs)
    assert mock_st.session_state["batch_id"] == 2

    mock_transcribe.side_effect = RuntimeError("Transcription produced no text")
    _handle_transcription([_make_file("third.mp3")], **kwargs)
    assert mock_st.session_state["batch_id"] == 3


# --- _transcription_kwargs ---


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"translate": True, "language": "fr"}, {"task": "translate"}),
        ({"translate": False, "language": "fr"}, {"task": "transcribe"}),
        ({"decode_independently": True}, {"condition_on_previous_text": False}),
        ({"decode_independently": False}, {"condition_on_previous_text": True}),
    ],
    ids=["translate_on", "translate_off", "no_context", "with_context"],
)
def test_transcription_kwargs_mappings(overrides, expected):
    kwargs = _transcription_kwargs(**_ui_state(**overrides))
    assert {k: kwargs[k] for k in expected} == expected


def test_transcription_kwargs_passes_through_unchanged_fields():
    kwargs = _transcription_kwargs(
        **_ui_state(
            language="en",
            include_subtitles=True,
            initial_prompt="hello",
            no_verbatim=True,
            clip_timestamps="30,90",
        )
    )
    assert kwargs["language"] == "en"
    assert kwargs["include_subtitles"] is True
    assert kwargs["initial_prompt"] == "hello"
    assert kwargs["no_verbatim"] is True
    assert kwargs["clip_timestamps"] == "30,90"


# --- _display_transcription ---


def test_display_transcription_no_session_state(mock_st):
    _display_transcription()
    mock_st.text_area.assert_not_called()


def test_display_transcription_shows_transcript(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        "Hello world",
        height=300,
        label_visibility="collapsed",
        key="transcript_b0_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_txt_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        "Hello world",
        "interview_transcript.txt",
        "text/plain",
        icon=":material/download:",
        key="download_txt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_srt_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(include_subtitles=True)]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        SRT_HELLO,
        "interview_transcript.srt",
        "application/x-subrip",
        icon=":material/download:",
        key="download_srt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_subtitles_on(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(include_subtitles=True)]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        SRT_HELLO,
        height=300,
        label_visibility="collapsed",
        key="transcript_b0_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_download_reflects_edits(mock_st):
    mock_st.text_area.side_effect = lambda label, value, **_: "edited transcript text"
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        "edited transcript text",
        "interview_transcript.txt",
        "text/plain",
        icon=":material/download:",
        key="download_txt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_right_aligns_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]
    _display_transcription()
    # "right", never "distribute": a standalone element in a distributed
    # container is left-aligned, which would silently unstick the shared edge.
    mock_st.container.assert_any_call(horizontal=True, horizontal_alignment="right")


def test_display_transcription_wraps_each_result_in_a_bordered_container(mock_st):
    mock_st.session_state["transcription"] = [
        _make_transcription(filename="first.mp3"),
        _make_transcription(filename="second.mp3"),
    ]

    _display_transcription()

    assert mock_st.container.call_args_list.count(call(border=True)) == 2


def test_display_transcription_multiple_files(mock_st):
    mock_st.session_state["transcription"] = [
        _make_transcription(file_stem="first_transcript", filename="first.mp3"),
        _make_transcription(file_stem="second_transcript", filename="second.mp3"),
    ]

    _display_transcription()

    assert mock_st.text_area.call_count == 2
    assert mock_st.download_button.call_count == 2
    assert mock_st.subheader.call_count == 2
    mock_st.subheader.assert_any_call("first.mp3")
    mock_st.subheader.assert_any_call("second.mp3")


def test_display_transcription_escapes_filename_in_subheader(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="my_song [live].mp3")]

    _display_transcription()

    mock_st.subheader.assert_called_once_with(r"my\_song \[live\].mp3")


def test_display_transcription_collapses_whitespace_in_subheader(mock_st):
    # st.subheader is NOT protected by the frontend's isLabel guard, contrary to
    # what a Markdown "label subset" implies: the heading component does
    # `[first, ...rest] = body.split("\n")` and renders `rest` through a bare
    # StreamlitMarkdown with no isLabel and no disallowedElements. So every line
    # after the first is full Markdown -- the same unguarded sink as st.error's
    # body. The raw name is what lands in the transcription dict, and only a
    # non-browser PUT to the upload route delivers a newline in one (see the
    # st.error case above), so this is defence in depth for browser clients and
    # a real guard for anything else.
    mock_st.session_state["transcription"] = [
        _make_transcription(filename="clip\n\n# Big\n\n> quote.mp3")
    ]

    _display_transcription()

    mock_st.subheader.assert_called_once_with("clip # Big > quote.mp3")


def test_display_transcription_keys_are_namespaced_by_batch(mock_st):
    mock_st.session_state["batch_id"] = 7
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    assert mock_st.text_area.call_args.kwargs["key"] == "transcript_b7_0"
    assert mock_st.download_button.call_args.kwargs["key"] == "download_txt_b7_0"


# --- formatting helpers ---


@pytest.mark.parametrize(
    "code,expected",
    [(None, "Detect"), ("en", "English"), ("fr", "French")],
    ids=["none_returns_detect", "lowercase_code", "title_cased"],
)
def test_format_language(code, expected):
    assert _format_language(code) == expected


@pytest.mark.parametrize(
    "count,expected",
    [(1, "1 file"), (2, "2 files"), (0, "0 files")],
    ids=["singular", "plural", "zero_is_plural"],
)
def test_plural(count, expected):
    assert _plural(count, "file") == expected


@pytest.mark.parametrize(
    "seconds,decimal_marker,expected",
    [
        (0.0, ".", "00:00:00.000"),
        (65.5, ".", "00:01:05.500"),
        (3661.123, ".", "01:01:01.123"),
        (65.5, ",", "00:01:05,500"),
    ],
    ids=["zero", "minutes_seconds", "hours", "comma_marker"],
)
def test_format_timestamp(seconds, decimal_marker, expected):
    assert _format_timestamp(seconds, decimal_marker=decimal_marker) == expected


def test_format_srt():
    assert _format_srt(MOCK_WHISPER_RESULT) == SRT_HELLO


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


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("interview.mp3", "audio/mpeg"),
        ("interview.m4a", "audio/mp4"),
        ("interview.wav", "audio/wav"),
        ("interview.opus", "audio/ogg"),
        ("clip.mp4", "video/mp4"),
        ("clip.mov", "video/quicktime"),
        ("clip.webm", "video/webm"),
        ("clip.mkv", "video/x-matroska"),
        ("SHOUTING.MP3", "audio/mpeg"),
        # Not in VIDEO_FORMATS, but Streamlit's normalize_upload_file_type pairs
        # `.mpeg4` with `.mp4` in the accept list, so the uploader takes it.
        ("clip.mpeg4", "video/mp4"),
        # Neither is reachable through the uploader, which enforces the accept
        # list server-side; they pin the fallback, which must stay a non-empty
        # audio type (the media route serves an empty mimetype as text/plain).
        ("mystery.xyz", "audio/wav"),
        ("download", "audio/wav"),
        # A dotfile named for an extension *is* accepted (`.mp3` ends with
        # `.mp3`) and Path.suffix reads it as extensionless -- the one upload
        # that lands on the fallback.
        (".mp3", "audio/wav"),
    ],
    ids=[
        "mp3",
        "m4a",
        "wav",
        "opus",
        "mp4",
        "mov",
        "webm",
        "mkv",
        "uppercase",
        "mpeg4_alias",
        "unknown_extension",
        "no_extension",
        "dotfile",
    ],
)
def test_media_mime(filename, expected):
    # st.audio's format= default is "audio/wav" for every input, and it becomes the
    # served Content-Type and the media URL's extension — not a hint.
    assert _media_mime(filename) == expected


def test_media_mime_covers_every_upload_format():
    # Compared against what the uploader *accepts*, not what the app declares:
    # normalize_upload_file_type is what st.file_uploader runs `type` through
    # before handing it to the frontend and to enforce_filename_restriction, and
    # it pairs `.mpeg4` in beside `.mp4` (TYPE_PAIRS). An extension in that list
    # without a MEDIA_MIME_TYPES entry falls through to DEFAULT_MEDIA_MIME and
    # ships the WAV mis-declaration for that one extension, with nothing else in
    # the suite to notice. Equality, not subset: the map used to carry extras
    # for the remote paths, and an entry the uploader can never reach is dead
    # weight that hides the next one.
    accepted = normalize_upload_file_type(AUDIO_FORMATS + VIDEO_FORMATS)
    assert set(MEDIA_MIME_TYPES) == {ext.lstrip(".") for ext in accepted}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("interview.mp3", "interview.mp3"),
        ("interview_part_1.mp3", r"interview\_part\_1.mp3"),
        ("Song [Official Video].mp3", r"Song \[Official Video\].mp3"),
        ("a*b`c~d:e$f&g", r"a\*b\`c\~d\:e\$f\&g"),
        (r"back\slash", r"back\\slash"),
        # `&` is escaped because micromark's characterReference is a *parse-time*
        # construct: unescaped, `&#58;` decodes to a colon early enough for the
        # frontend's post-parse pass to turn `:streamlit:` into the logo image.
        ("Rock &amp; Roll.mp3", r"Rock \&amp; Roll.mp3"),
        ("clip&#58;streamlit&#58;.mp3", r"clip\&#58;streamlit\&#58;.mp3"),
        # Block markers are defused by losing the line start, not by escaping --
        # `#` and `>` come back literal. Both sinks that render without the
        # frontend's isLabel flag (st.error's body and st.subheader's lines after
        # the first) depend on this, so it lives here rather than at a call site.
        ("clip\n\n# Big\n\n> quote.mp3", "clip # Big > quote.mp3"),
        ("spaced   out\ttabbed.mp3", "spaced out tabbed.mp3"),
    ],
    ids=[
        "plain",
        "underscores",
        "brackets",
        "all_specials",
        "backslash",
        "named_entity",
        "numeric_entity",
        "block_markers",
        "whitespace_runs",
    ],
)
def test_escape_markdown(raw, expected):
    assert _escape_markdown(raw) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Short line.", "Short line."),
        (
            "The quick brown fox jumps over the lazy dog and keeps on running.",
            "The quick brown fox jumps over the\nlazy dog and keeps on running.",
        ),
        # An embedded newline would otherwise close the cue early and corrupt
        # every cue after it, so the whitespace collapse is structural.
        ("line one\n\nline two", "line one line two"),
        ("  padded   out  ", "padded out"),
    ],
    ids=["short_passthrough", "balanced_two_lines", "embedded_newlines", "whitespace"],
)
def test_wrap_cue(text, expected):
    assert _wrap_cue(text) == expected


def test_wrap_cue_balances_instead_of_orphaning():
    # Greedy wrapping fills each line to the brim and leaves a short tail, which
    # is more conspicuous over a picture than the long line it replaced. The
    # re-wrap at the narrowest width holding the same line count is what fixes it.
    #
    # This sentence is chosen, not invented: plain textwrap gives 34/11 here and
    # the balanced pass gives 24/21. Most sentences wrap identically under both,
    # so an arbitrary long string makes this assertion pass without testing
    # anything -- the first draft of this test used one and did exactly that.
    lines = _wrap_cue("The internationalization committee reconvened.").split("\n")

    assert len(lines) == 2
    assert all(len(line) <= SUBTITLE_LINE_WIDTH for line in lines)
    assert abs(len(lines[0]) - len(lines[1])) <= 5


def test_wrap_cue_never_splits_a_token():
    # break_long_words=False is not enough on its own: textwrap splits on hyphens
    # by default, which cut this URL in half mid-path. An unbreakable token now
    # overflows onto its own line instead, which is the accepted tradeoff.
    url = "https://example.com/an-extremely-long-path-that-cannot-be-broken"
    lines = _wrap_cue(f"Visit {url} now").split("\n")

    assert url in lines
    assert "".join(lines).count("-") == url.count("-")


def test_format_srt_drops_text_less_cues():
    # mlx *clears* rather than removes a blank segment -- "if a segment is
    # instantaneous or does not contain text, clear it" (transcribe.py) sets text
    # to "" and still extends all_segments with it. Emitting one yields a block
    # with an index, a timing line and no text: malformed SRT. Reachable on any
    # run, since the same branch fires whenever start == end.
    srt = _format_srt(
        {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " Hello there"},
                {"start": 2.0, "end": 5.0, "text": ""},
                {"start": 5.0, "end": 5.0, "text": "   "},
                {"start": 7.0, "end": 9.0, "text": " Goodbye"},
            ]
        }
    )

    blocks = srt.strip("\n").split("\n\n")
    assert len(blocks) == 2
    # Numbering must close the gap, not preserve the dropped indices.
    assert [b.split("\n")[0] for b in blocks] == ["1", "2"]
    assert blocks[1].endswith("Goodbye")


def test_format_srt_wraps_long_cues():
    # The cue is valid SRT unwrapped and every player accepts it -- this is a
    # display convention, so nothing but an explicit assertion catches a regression.
    long_text = "A cue whose text runs well past the forty-two character line limit easily."
    srt = _format_srt({"segments": [{"start": 0.0, "end": 3.0, "text": f" {long_text}"}]})

    index, timing, *cue_lines = srt.rstrip("\n").split("\n")
    assert index == "1"
    assert timing == "00:00:00,000 --> 00:00:03,000"
    assert len(cue_lines) > 1
    assert all(len(line) <= SUBTITLE_LINE_WIDTH for line in cue_lines)
    assert " ".join(cue_lines) == long_text


@pytest.mark.parametrize(
    "raw",
    ["", "30,90", "0,60,120,180", " 30 , 90 ", "0,0.5", "1.5,2", "30", "30,60,90"],
    ids=[
        "blank",
        "single",
        "multi",
        "whitespace",
        "decimal_start",
        "decimal_pair",
        "trailing_start",
        "trailing_start_multi",
    ],
)
def test_validate_time_range_valid(raw):
    assert _validate_time_range(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "abc",
        "30,abc",
        "-5,10",
        "90,30",
        "30,30",
        "30,",
        ",30",
        "60,90,0,30",
        # float() accepts all four of these, and every comparison against a nan is
        # False -- so they passed validation, enabled Transcribe, and failed
        # inside mlx_whisper instead. "1e400" overflows to inf on the way in.
        "nan",
        "inf",
        "0,inf",
        "1e400",
    ],
    ids=[
        "non_numeric",
        "non_numeric_pair",
        "negative",
        "end_before_start",
        "equal_pair",
        "trailing_comma",
        "leading_comma",
        "out_of_order",
        "nan",
        "inf",
        "inf_end",
        "overflow_literal",
    ],
)
def test_validate_time_range_invalid(raw):
    assert _validate_time_range(raw) is not None


# --- module UI (AppTest) ---
#
# These exercise the module-level UI (page config, tabs, buttons, fragment) by
# running the real script through Streamlit's AppTest runtime, complementing the
# mocked-`st` unit tests above.


APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def _run_app(transcription=None, active_tab=None):
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    if transcription is not None:
        at.session_state["transcription"] = transcription
    if active_tab is not None:
        # AppTest has no tab-selection API; st.tabs' `key` holds the active label.
        at.session_state["input_tabs"] = active_tab
    return at.run()


def test_page_config():
    assert PAGE_CONFIG == {
        "page_title": "Whisper Transcribe",
        "page_icon": ":material/graphic_eq:",
        "layout": "centered",
    }


def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception
    assert [t.value for t in at.title] == ["Whisper Transcribe"]


def test_tabs_have_material_icon_labels():
    at = _run_app()
    assert [t.label for t in at.tabs] == [
        ":material/upload: Upload",
        ":material/mic: Record",
    ]


def test_transcribe_button_has_icon_and_is_disabled_without_audio():
    button = _run_app().button[0]
    assert button.label == "Transcribe"
    assert button.icon == ":material/graphic_eq:"
    assert button.disabled is True


def test_invalid_time_range_shows_inline_error():
    at = _run_app()
    next(t for t in at.text_input if t.label == "Time range").set_value("90,30").run()
    assert [e.value for e in at.error] == [_validate_time_range("90,30")]


def test_valid_time_range_shows_no_error():
    at = _run_app()
    next(t for t in at.text_input if t.label == "Time range").set_value("30,90").run()
    assert at.error == []


def test_results_render_download_button_with_icon():
    # Seeded results render through the st.fragment(_display_transcription)() wrap.
    at = _run_app([_make_transcription()])
    assert not at.exception
    assert [s.value for s in at.subheader] == ["interview.mp3"]
    assert at.text_area[0].value == "Hello world"
    download = at.get("download_button")[0]
    assert download.label == "Download"
    assert download.icon == ":material/download:"


def test_no_results_renders_no_download_button():
    assert _run_app().get("download_button") == []


def _publish(at, transcription, batch):
    # Mirrors _handle_transcription's publish: the results plus the batch id
    # that namespaces the transcript widget keys.
    at.session_state["transcription"] = transcription
    at.session_state["batch_id"] = batch
    return at.run()


def test_new_batch_replaces_previous_transcript_text():
    # A keyed st.text_area restores its session-state value and ignores the
    # `value` argument, so a batch-invariant key renders the *previous* batch's
    # text under the new filename -- and the Download button, whose payload is
    # the text area's return value, serves it. This needs two renders with
    # different data; no single-render test can see it.
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    _publish(at, [_make_transcription(filename="first.mp3")], batch=1)
    assert at.text_area[0].value == "Hello world"

    # Edits must still stick *within* a batch -- that is the point of the key.
    at.text_area[0].set_value("edited by hand").run()
    assert at.text_area[0].value == "edited by hand"

    _publish(at, [_make_transcription(filename="second.mp3", text="Second file text")], batch=2)
    assert not at.exception
    assert [s.value for s in at.subheader] == ["second.mp3"]
    assert at.text_area[0].value == "Second file text"


def test_new_batch_replaces_previous_transcript_when_subtitles_toggled():
    # Flipping include_subtitles changes only the *download* key (txt -> srt),
    # so without the batch namespace the transcript text area stays stale and
    # the SRT cues never reach the screen.
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    _publish(at, [_make_transcription(filename="first.mp3")], batch=1)
    assert at.text_area[0].value == "Hello world"

    _publish(
        at,
        [
            _make_transcription(
                filename="second.mp3", text="Second file text", include_subtitles=True
            )
        ],
        batch=2,
    )
    assert not at.exception
    assert at.text_area[0].value == "1\n00:00:00,000 --> 00:00:02,500\nSecond file text\n"


UPLOAD_TAB = ":material/upload: Upload"
RECORD_TAB = ":material/mic: Record"


def _tab(at, label):
    return next(t for t in at.tabs if t.label == label)


def _upload(at, name="upload.mp3", data=b"upload bytes", mime="audio/mpeg"):
    # FileUploader.set_value takes (name, bytes, mime), or a sequence of those
    # under accept_multiple_files=True, since at least the 1.59 floor (checked
    # directly against 1.59.0). It is the one source AppTest's element tree can
    # seed: there is no audio_input element in testing/v1, and st.audio_input
    # registers with writes_allowed=False, so session_state cannot plant one.
    # See _recording for the route that can.
    at.file_uploader[0].set_value([(name, data, mime)])
    return at.run()


def _recording(name="recording.wav", data=b"wav bytes"):
    # A recording for AppTest, planted by patching `streamlit.audio_input` around
    # the run -- the same module-level route the suite already takes for
    # `mlx_whisper.transcribe`, and the only one that works (see _upload). The
    # script does `st.audio_input(...)` at call time, so the patched attribute is
    # what it reaches; no widget is registered, which nothing here depends on.
    rec = UploadedFile(UploadedFileRec("id", name, "audio/wav", data), FileURLs())
    return patch("streamlit.audio_input", return_value=rec)


def _assert_declared_mime(element, mime):
    """Assert an st.audio element was given `format=mime`, via its media URL.

    The mimetype is not in the proto, but Streamlit derives the URL's extension
    from it, so the suffix is the only observable proxy. Both sides are computed
    with Streamlit's own lookup rather than hardcoded, because that lookup is
    `mimetypes.guess_extension` -- the same platform-varying table MEDIA_MIME_TYPES
    exists to avoid. The inequality guard keeps the assertion from passing
    vacuously if a host's tables map both types to the same (or an empty) suffix.
    """
    expected = get_extension_for_mimetype(mime)
    assert expected and expected != get_extension_for_mimetype(DEFAULT_MEDIA_MIME)
    assert element.proto.url.endswith(expected)


def test_upload_preview_renders_with_its_declared_mime():
    at = _upload(_run_app(), name="clip.mp3", mime="audio/mpeg")
    assert not at.exception
    previews = _tab(at, UPLOAD_TAB).get("audio")
    assert len(previews) == 1
    # Pins the format= wiring: test_media_mime covers *which* mimetype is chosen,
    # but only a rendered element shows it was passed at all. Dropping format=
    # from the st.audio call serves every preview as .wav again, and nothing
    # else in the suite can see that.
    _assert_declared_mime(previews[0], "audio/mpeg")


def test_upload_enables_transcribe():
    at = _upload(_run_app())
    assert at.button[0].disabled is False


def test_transcription_failure_renders_an_escaped_alert():
    # The one case that pushes an _error message through the *real* st.error
    # rather than mock_st: a per-file RuntimeError must surface as an alert on
    # the page, escaped, with no traceback. `assert not at.exception` is the
    # discriminating half -- the unexpected-exception branch attaches one via
    # st.exception, so a message-only check would not separate the two.
    with patch("mlx_whisper.transcribe", side_effect=RuntimeError("Failed to load audio")):
        at = _upload(_run_app(), name="my_clip.mp3")
        at.button[0].click().run()

    assert not at.exception
    assert [e.value for e in at.error] == [
        r"Transcription failed for my\_clip.mp3\: Failed to load audio"
    ]
    assert at.session_state["transcription"] == []


@pytest.mark.parametrize(
    "tab_sources,expected",
    [
        ([(False, ["upload"]), (True, ["recording"])], ["recording"]),
        ([(False, ["upload"]), (True, [])], ["upload"]),
        ([(False, ["upload"]), (False, ["recording"])], ["upload"]),
        ([(None, ["upload"]), (None, ["recording"])], ["upload"]),
        ([(True, []), (False, [])], []),
    ],
    ids=[
        "open_tab_wins",
        "empty_open_tab_falls_back",
        "no_open_tab_falls_back_in_priority_order",
        "untracked_tabs_fall_back",
        "nothing_loaded",
    ],
)
def test_active_sources(tab_sources, expected):
    # `open_tab_wins` is the only case that fails when the `is_open and` filter
    # is dropped. `untracked_tabs_fall_back` feeds the None that TabContainer.open
    # returns without on_change="rerun"; with *every* flag None, "None is closed"
    # and "None is open" both yield the priority order, so it pins only that None
    # is accepted -- the semantics are unobservable from outside, by construction.
    # The script-level check that the body routes through this helper at all is
    # test_transcribe_runs_the_open_tab_over_a_loaded_upload.
    assert _active_sources(tab_sources) == expected


def test_transcribe_runs_the_open_tab_over_a_loaded_upload():
    # The end-to-end mutation check for the dispatch. Record tab open with a
    # recording planted (see _recording) *and* an upload still loaded: the
    # recording must win. Fails under either regression the unit test above
    # cannot see -- replacing the _active_sources call with a flat
    # `uploaded_files or [recorded_audio]` chain, or dropping on_change="rerun"
    # from st.tabs so every `.open` reads None and the helper falls back.
    with _recording(), patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app(active_tab=RECORD_TAB))
        at.button[0].click().run()

    assert not at.exception
    assert [d["filename"] for d in at.session_state["transcription"]] == ["recording.wav"]


def test_transcribe_falls_back_to_a_loaded_upload_from_an_empty_tab():
    # Record tab open with nothing recorded, an upload still loaded: the fallback
    # in _active_sources runs the upload rather than leaving Transcribe dead. This
    # pins the fallback *only* -- it passes with or without the "open tab wins"
    # filter, by design, so a later "only the open tab counts" simplification
    # cannot silently make Transcribe dead. The winning half is the case above.
    with patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app(active_tab=RECORD_TAB))
        at.button[0].click().run()

    assert not at.exception
    assert [d["filename"] for d in at.session_state["transcription"]] == ["upload.mp3"]


def test_transcript_format_defaults_to_plain_text():
    # This pins `default=` only. The companion `required=True` is deliberately NOT
    # asserted here because AppTest cannot see it: per the docstring it stops a user
    # from deselecting the chosen option in the browser ("clicking an already-selected
    # option does nothing"), and AppTest's ButtonGroup.unselect() is a no-op for a
    # single-select group whether or not required is set -- verified by deleting
    # required=True and re-running, which changes nothing observable. So dropping
    # required=True is a silent regression as far as this suite is concerned: it would
    # let the widget return None, which reads as plain text through the
    # `== FORMAT_SUBTITLES` comparison while looking like nothing is selected.
    at = _run_app()
    assert at.segmented_control[0].value == FORMAT_PLAIN_TEXT


@pytest.mark.parametrize(
    "choice,expected",
    [(FORMAT_PLAIN_TEXT, False), (FORMAT_SUBTITLES, True)],
    ids=["plain_text", "subtitles"],
)
def test_transcript_format_drives_include_subtitles(choice, expected):
    # `include_subtitles = transcript_format == FORMAT_SUBTITLES` is a module-level
    # comparison, so the only place the mapping is observable is what
    # _handle_transcription stores. Drive the whole path: an upload to enable
    # Transcribe, a stubbed model, then read the recorded flag back. An inverted
    # comparison would otherwise ship silently -- it changes no widget state, only
    # the download's extension and the text area's contents.
    with patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app())
        at.segmented_control[0].set_value(choice).run()
        at.button[0].click().run()

    assert not at.exception
    assert at.session_state["transcription"][0]["include_subtitles"] is expected
