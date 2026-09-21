import re
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import streamlit as st
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError
from streamlit.elements.lib.file_uploader_utils import normalize_upload_file_type
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.memory_media_file_storage import get_extension_for_mimetype
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, UnknownElement

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    BUTTON_WIDTH,
    DEFAULT_MEDIA_MIME,
    EMPTY_RESULTS_HINT,
    ERROR_ICON,
    ERROR_MESSAGE_LIMIT,
    FORMAT_PLAIN_TEXT,
    FORMAT_SUBTITLES,
    MEDIA_MIME_TYPES,
    PAGE_CONFIG,
    SUBTITLE_LINE_WIDTH,
    TRANSCRIPT_FORMATS,
    TRANSCRIPT_HEIGHT,
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

# Spelled out: this is the pin, so importing RESULT_ICON would make it tautological.
RESULT_HEADING = ":material/description: {}"

# Annotated so ty reads result["text"] as Any rather than the literal's union.
MOCK_WHISPER_RESULT: dict[str, Any] = {
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


def _status(mock_st):
    # The handle _handle_transcription's `with st.status(...) as status:` binds.
    return mock_st.status.return_value.__enter__.return_value


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
    result: dict[str, Any] = MOCK_WHISPER_RESULT
    if text is not None:
        # Only the keys _format_srt and the transcript expression below read.
        result = {"text": text, "segments": [{"start": 0.0, "end": 2.5, "text": text}]}
    return {
        # Mirrors _handle_transcription, which renders the text once at publish
        # time and stores nothing of the raw mlx result.
        "transcript": _format_srt(result) if include_subtitles else result["text"].strip(),
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
    # Nothing is @st.cache_resource today. Kept anyway: it is a no-op on an
    # empty store, the two APIs own separate singletons (measured -- clearing
    # one leaves the other cached), and the first cache_resource function added
    # back would otherwise re-create the order-dependence above with nothing to
    # fail until someone remembered this line.
    st.cache_resource.clear()


DOWNLOAD_LABEL = "Downloading model weights (first run only)..."


@pytest.fixture(autouse=True)
def mock_snapshot_download():
    # _handle_transcription probes the weight cache through
    # huggingface_hub.snapshot_download before its loop. Patched for every test,
    # as a cache hit: otherwise the ~20 mocked _handle_transcription calls and
    # every AppTest click would reach the Hub on each Stop-hook pytest, and the
    # macos-14 runner, which has no cache, would download 1.5 GB. The module
    # attribute is the target because the app calls it that way, so the one
    # patch covers the imported module and the AppTest re-execution alike.
    with patch("huggingface_hub.snapshot_download") as m:
        yield m


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


def test_transcript_height():
    # Measured, not picked: keeps a one-file result (status, heading, text area,
    # Download) inside a 1920x1080 display's 839px Chrome viewport with 66px to
    # spare; 400 leaves 26px. Pinned literally because every other assertion on
    # the height compares against the constant itself.
    assert TRANSCRIPT_HEIGHT == 360


def test_empty_results_hint():
    assert EMPTY_RESULTS_HINT == ":material/subtitles: Transcripts appear here"


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
    # `white-space: nowrap; text-overflow: ellipsis` line, and the whole hint
    # measures 357px at Source Sans 14px (92px for the size prefix, ~265px for
    # the list; each entry costs ~31-40px). Under the old `centered` layout the
    # span had a fixed 571px at any desktop width. It is fluid now -- the input
    # column is half the main area -- so this length proxy guards the list, not
    # the viewport: with the sidebar open the slot the span fills is 587px at
    # 1920 and 357px at 1460, and from 1458 down the list loses entries to an
    # ellipsis (MKV at 1440; measured in Chrome, with the theme's 1px widget
    # border on the dropzone -- stock's borderless dropzone fit down to 1456).
    # No test can see that half; CLAUDE.md "Accepted formats" carries the
    # numbers.
    hint = ", ".join(f.upper() for f in AUDIO_FORMATS + VIDEO_FORMATS)
    assert len(hint) <= 60, f"{hint!r} will truncate in the uploader dropzone"


# --- theme (.streamlit/config.toml) ---

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
THEME_MODES = ("light", "dark")
_SEMANTIC = ("red", "orange", "yellow", "green", "blue", "violet", "gray")
# Every [theme] key that exists at the streamlit>=1.59 floor, read from
# `streamlit.config._config_options_template` under a 1.59.0 install: 52 keys on
# the top-level table, and the same 42 on each of [theme.light], [theme.dark],
# [theme.sidebar] and the two [theme.<mode>.sidebar] tables -- the ten in
# THEME_TOP_LEVEL_ONLY_KEYS are accepted nowhere else. This is what Streamlit
# *reads*; what the theme is *allowed* to set (no typography) is the separate
# THEME_TYPOGRAPHY_KEYS policy and its own test.
THEME_KEYS_AT_FLOOR = frozenset(
    {
        "base",
        "primaryColor",
        "backgroundColor",
        "secondaryBackgroundColor",
        "textColor",
        "borderColor",
        "linkColor",
        "linkUnderline",
        "codeBackgroundColor",
        "codeTextColor",
        "dataframeBorderColor",
        "dataframeHeaderBackgroundColor",
        "baseRadius",
        "buttonRadius",
        "showWidgetBorder",
        "showSidebarBorder",
        "font",
        "headingFont",
        "codeFont",
        "fontFaces",
        "baseFontSize",
        "baseFontWeight",
        "codeFontSize",
        "codeFontWeight",
        "headingFontSizes",
        "headingFontWeights",
        "metricValueFontSize",
        "metricValueFontWeight",
        "chartCategoricalColors",
        "chartSequentialColors",
        "chartDivergingColors",
    }
    | {f"{c}{suffix}" for c in _SEMANTIC for suffix in ("Color", "BackgroundColor", "TextColor")}
)
THEME_TOP_LEVEL_ONLY_KEYS = frozenset(
    {
        "base",
        "baseFontSize",
        "baseFontWeight",
        "fontFaces",
        "metricValueFontSize",
        "metricValueFontWeight",
        "showSidebarBorder",
        "chartCategoricalColors",
        "chartSequentialColors",
        "chartDivergingColors",
    }
)
THEME_TABLE_KEYS_AT_FLOOR = {
    "theme": THEME_KEYS_AT_FLOOR,
    **{
        name: THEME_KEYS_AT_FLOOR - THEME_TOP_LEVEL_ONLY_KEYS
        for name in ("theme.light", "theme.dark", "theme.sidebar")
        + tuple(f"theme.{m}.sidebar" for m in THEME_MODES)
    },
}
# Layout constants, not colours: every pixel measurement in CLAUDE.md (button
# widths, column widths, the dropzone hint's 357px, the sidebar label that was
# renamed because it wrapped) depends on the bundled Source Sans at 16px, and a
# Google Fonts URL would call out from an app that promises to run locally.
THEME_TYPOGRAPHY_KEYS = frozenset(
    {
        "font",
        "headingFont",
        "codeFont",
        "fontFaces",
        "baseFontSize",
        "baseFontWeight",
        "codeFontSize",
        "codeFontWeight",
        "headingFontSizes",
        "headingFontWeights",
        "metricValueFontSize",
        "metricValueFontWeight",
    }
)
# Streamlit's stock primary, red in both modes, and its stock redColor per mode
# (one step darker in dark). The collision the theme exists to fix is that the
# primary and the error red are the same hue.
STOCK_PRIMARY = "#ff4b4b"
STOCK_RED = {"light": "#ff4b4b", "dark": "#ff2b2b"}


def _config():
    return tomllib.loads(CONFIG_PATH.read_text())


def _theme():
    return _config()["theme"]


def test_usage_stats_are_off():
    # Read from the top-level table, not through _theme(): nothing else in the
    # gate can see the config, so a dropped line would ship telemetry under a
    # green gate. Why it is off: the comment above the key in config.toml.
    assert _config()["browser"]["gatherUsageStats"] is False


def _scalars(table):
    return {k: v for k, v in table.items() if not isinstance(v, dict)}


def _theme_tables(table, path="theme"):
    """Yield (path, scalar entries) for [theme] and every nested table under it.

    Walks every dict-valued entry rather than a fixed list of names, so a typo'd
    subsection ([theme.dark.sidebr]) reaches the key check instead of being
    skipped.
    """
    yield path, _scalars(table)
    for key, value in table.items():
        if isinstance(value, dict):
            yield from _theme_tables(value, f"{path}.{key}")


def _mode_colours(theme, mode):
    # Streamlit merges the top-level [theme] table into each mode, so a colour set
    # once under [theme] applies to both; model the merge rather than require the
    # colour to be repeated per mode.
    return {**_scalars(theme), **_scalars(theme.get(mode, {}))}


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)


def _luminance(hex_color):
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in _rgb(hex_color)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _hsl(hex_color):
    r, g, b = _rgb(hex_color)
    high, low = max(r, g, b), min(r, g, b)
    lightness = (high + low) / 2
    if high == low:
        return None, 0.0, lightness
    d = high - low
    saturation = d / (2 - high - low) if lightness > 0.5 else d / (high + low)
    if high == r:
        hue = ((g - b) / d) % 6
    elif high == g:
        hue = (b - r) / d + 2
    else:
        hue = (r - g) / d + 4
    return (hue * 60) % 360, saturation, lightness


def _shift_lightness(hex_color, delta):
    """Streamlit's lighten/darken: HSL lightness +- delta on a 0-1 scale, emitted as
    hsla() with the hue rounded to whole degrees and s/l to whole percent."""
    hue, saturation, lightness = _hsl(hex_color)
    hue = round(hue or 0) % 360
    saturation, lightness = (
        round(saturation * 100) / 100,
        round(min(1, max(0, lightness + delta)) * 100) / 100,
    )
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    x = chroma * (1 - abs((hue / 60) % 2 - 1))
    sector = int(hue // 60) % 6
    r, g, b = [
        (chroma, x, 0),
        (x, chroma, 0),
        (0, chroma, x),
        (0, x, chroma),
        (x, 0, chroma),
        (chroma, 0, x),
    ][sector]
    m = lightness - chroma / 2
    return _hex((r + m, g + m, b + m))


def _over(hex_color, alpha, hex_background):
    fg, bg = _rgb(hex_color), _rgb(hex_background)
    return _hex(tuple(alpha * fg[i] + (1 - alpha) * bg[i] for i in range(3)))


def _hue_distance(a, b):
    ha, hb = _hsl(a)[0], _hsl(b)[0]
    assert ha is not None and hb is not None, "achromatic primary or red"
    d = abs(ha - hb)
    return min(d, 360 - d)


def _rendered_alert(colours, mode):
    """The pair st.error paints: redTextColor on redBackgroundColor, both derived
    from redColor when not set explicitly -- the frontend lightens the red 15% on
    a dark page and darkens it 15% on a light one (light = page luminance > 0.5),
    and tints the box with the red at 20% (dark) / 10% (light) over the page."""
    page = colours["backgroundColor"]
    red = colours.get("redColor", STOCK_RED[mode])
    light = _luminance(page) > 0.5
    text = colours.get("redTextColor", _shift_lightness(red, -0.15 if light else 0.15))
    box = colours.get("redBackgroundColor", _over(red, 0.1 if light else 0.2, page))
    return text, box


def test_theme_defines_both_modes_and_colours_them_completely():
    # The frontend keeps the System / Light / Dark switcher as long as *either*
    # [theme.light] or [theme.dark] carries a value; this repo's rule is stricter
    # -- both defined, each resolving the four base colours after the [theme]
    # merge -- so one mode can never ship Streamlit's stock palette beside the
    # themed other.
    theme = _theme()
    for mode in THEME_MODES:
        assert _scalars(theme.get(mode, {})), f"[theme.{mode}] is missing or empty"
        assert {"primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"} <= set(
            _mode_colours(theme, mode)
        )


def test_theme_uses_only_floor_keys_and_six_digit_hex():
    # An unknown key -- or a known key in a table Streamlit does not read it from
    # -- is logged as a warning and dropped, so a typo would ship stock under a
    # green gate; an unknown *table* is dropped the same way. An invalid colour
    # string is quieter still: the server accepts it without a word and only the
    # browser console drops it, so the #rrggbb check below is the sole guard --
    # and names or rgba() would break the contrast arithmetic in any case.
    for name, table in _theme_tables(_theme()):
        allowed = THEME_TABLE_KEYS_AT_FLOOR.get(name)
        assert allowed is not None, f"[{name}] is not a table Streamlit reads"
        for key, value in table.items():
            assert key in allowed, f"[{name}] {key} is not read from that table at 1.59.0"
            if key.endswith("Color"):
                assert HEX_COLOR.match(value), f"[{name}] {key} = {value!r} is not #rrggbb"


def test_theme_sets_no_typography():
    for name, table in _theme_tables(_theme()):
        assert not (set(table) & THEME_TYPOGRAPHY_KEYS), f"[{name}] sets a font or size key"


@pytest.mark.parametrize("mode", THEME_MODES)
def test_theme_contrast(mode):
    colours = _mode_colours(_theme(), mode)
    page, well = colours["backgroundColor"], colours["secondaryBackgroundColor"]
    text, primary = colours["textColor"], colours["primaryColor"]
    # Body text sits on both backgrounds: the page, and the text area / dropzone /
    # selectbox wells on the secondary.
    assert _contrast(text, page) >= 4.5
    assert _contrast(text, well) >= 4.5
    # primaryColor is also *text*: the active tab label sits on the page, and the
    # selected Transcript format option sits in the sidebar -- whose background is
    # the secondary colour -- on a cell tinted 10% primary. That second pair cannot
    # reach 4.5 with a primary that also holds a white button label (see below),
    # so it is held at the 3:1 component floor; the tab label carries the text
    # standard.
    assert _contrast(primary, page) >= 4.5
    assert _contrast(primary, _over(primary, 0.1, well)) >= 3.0
    # The primary button always paints a white label. No blue can put that label
    # at 4.5 while primary-as-text stays at 4.5 on a page brighter than #040507
    # (the two bounds cross at luminance 0.183 vs 0.227); the theme takes the
    # text side and holds the label at the 3:1 large-text/component floor, up
    # from stock's 3.3, and hover darkens the fill, which lifts it past 6.
    assert _contrast("#ffffff", primary) >= 3.0
    # st.error paints the *derived* red text on the *derived* tint, never the raw
    # redColor on the page; the status widget's error icon paints the same derived
    # text on the page.
    alert_text, alert_box = _rendered_alert(colours, mode)
    assert _contrast(alert_text, alert_box) >= 4.5
    assert _contrast(alert_text, page) >= 4.5
    # Hairline borders by design (Apple separators, not 3:1 boundaries), but a
    # border that vanished into either surface would drop every field edge at
    # once -- and most bordered fields (selectbox, text input, expander, the
    # Keyterms multiselect) sit on the sidebar, whose background is the well.
    # Unset, Streamlit derives the border from textColor at 20% alpha, which the
    # config's shed order names as the first thing to drop; nothing to check then.
    if "borderColor" in colours:
        assert _contrast(colours["borderColor"], page) >= 1.5
        assert _contrast(colours["borderColor"], well) >= 1.5


@pytest.mark.parametrize("mode", THEME_MODES)
def test_theme_primary_is_not_the_error_hue(mode):
    # The defect the theme fixes: stock's primary and its error red are the same
    # hue (distance 0), so Transcribe, the active tab and the chosen format read
    # as alerts. Reverting primaryColor to the stock red fails here and only here.
    colours = _mode_colours(_theme(), mode)
    assert _hue_distance(colours["primaryColor"], colours.get("redColor", STOCK_RED[mode])) >= 90


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
    assert set(data) == {"transcript", "file_stem", "filename", "include_subtitles"}
    assert data["transcript"] == "Hello world"
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
    data = mock_st.session_state["transcription"][0]
    assert data["include_subtitles"] is True
    # The flag decides the rendered text at publish time, not at display time.
    assert data["transcript"] == SRT_HELLO


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

    status = _status(mock_st)
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

    # Positional label only: expanded=True was dropped as inert (nothing renders
    # inside the block, and the first per-file update clears the field), so a
    # re-added one fails here and has to say what it is for.
    mock_st.status.assert_called_once_with(opening)
    status = _status(mock_st)
    status.update.assert_called_with(label=closing, state="complete")


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_probes_the_weight_cache_offline(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # The common path: weights cached, so the probe is one offline call and the
    # first-run label never renders. local_files_only=True is the whole point --
    # a bare snapshot_download resolves `main` against the Hub even when cached
    # (two GETs per click) and would flash a false "first run" label every time.
    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_snapshot_download.assert_called_once_with(repo_id=ASR_MODEL_REPO, local_files_only=True)
    status = _status(mock_st)
    assert call(label=DOWNLOAD_LABEL) not in status.update.call_args_list


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_labels_the_first_run_download(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # On a cache miss the label switches *before* the fetch -- the point of the
    # feature, so the fetch stub reads the labels set so far at the moment it
    # runs; an index comparison against the per-file label would pass with the
    # update moved after the fetch. The fetch is the same
    # snapshot_download(repo_id=...) mlx_whisper's load_model makes, so the
    # model's own call then finds the weights cached.
    status = _status(mock_st)
    labels_at_fetch = []

    def fetch(repo_id, **kwargs):
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("miss")
        labels_at_fetch.append([c.kwargs.get("label") for c in status.update.call_args_list])

    mock_snapshot_download.side_effect = fetch

    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    assert mock_snapshot_download.call_args_list == [
        call(repo_id=ASR_MODEL_REPO, local_files_only=True),
        call(repo_id=ASR_MODEL_REPO),
    ]
    assert labels_at_fetch == [[DOWNLOAD_LABEL]]
    mock_transcribe.assert_called_once()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_labels_a_resumed_partial_download(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # An interrupted first run leaves a partial snapshot on disk. From
    # huggingface-hub 1.22 the offline probe reports it as
    # IncompleteSnapshotError, a LocalEntryNotFoundError subclass, so the
    # resumed download is labelled too; below 1.22 it read as a hit and the
    # resume ran unlabelled, which is why that version is the declared floor.
    mock_snapshot_download.side_effect = [
        IncompleteSnapshotError("partial", snapshot_path="/partial"),
        None,
    ]

    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    assert call(label=DOWNLOAD_LABEL) in _status(mock_st).update.call_args_list
    mock_transcribe.assert_called_once()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_leaves_a_failed_prefetch_to_the_loop(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # The prefetch is best-effort. If it fails (no network on a first run) the
    # batch must still run: mlx_whisper retries the download inside its own
    # call and *that* failure is caught per file and replayed as an alert, as it
    # was before the probe existed. An exception escaping here instead would
    # skip the replay and leave the status stuck on the download label.
    mock_snapshot_download.side_effect = [LocalEntryNotFoundError("miss"), OSError("offline")]

    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_transcribe.assert_called_once()
    status = _status(mock_st)
    status.update.assert_called_with(label="Transcribed 1/1 file", state="complete")
    mock_st.error.assert_not_called()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_keeps_the_previous_batch_when_the_prefetch_is_interrupted(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # The [] publish and the batch_id bump sit *after* the probe. A rerun request
    # that lands during the first-run download (a tab switch, any widget) is
    # raised at the next yield point -- that publish -- as RerunException, a
    # BaseException; with the publish first, it had already wiped the previous
    # batch, so the interrupted first run left an empty results column.
    class _Interrupt(BaseException):
        pass

    mock_st.session_state["transcription"] = [_make_transcription(filename="previous.mp3")]
    mock_st.session_state["batch_id"] = 3
    mock_snapshot_download.side_effect = [LocalEntryNotFoundError("miss"), _Interrupt()]

    with pytest.raises(_Interrupt):
        _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    assert [d["filename"] for d in mock_st.session_state["transcription"]] == ["previous.mp3"]
    assert mock_st.session_state["batch_id"] == 3
    mock_transcribe.assert_not_called()


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_ignores_a_probe_failure_that_is_not_a_miss(
    mock_transcribe, mock_st, mock_snapshot_download
):
    # Anything but LocalEntryNotFoundError out of the probe -- an unreadable ref,
    # a repo id the Hub validator rejects -- is not a miss: no label, no
    # prefetch, and the batch runs as it did before the probe existed. Escaping
    # the status block instead would skip the failure replay entirely.
    mock_snapshot_download.side_effect = OSError("unreadable ref")

    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_snapshot_download.assert_called_once_with(repo_id=ASR_MODEL_REPO, local_files_only=True)
    status = _status(mock_st)
    assert call(label=DOWNLOAD_LABEL) not in status.update.call_args_list
    status.update.assert_called_with(label="Transcribed 1/1 file", state="complete")
    mock_transcribe.assert_called_once()
    mock_st.error.assert_not_called()


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
    status = _status(mock_st)
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
    # A results renderer and nothing more: the empty-state hint is decided at
    # the call site (see test_empty_results_column_shows_the_hint_until_a_result_exists),
    # so with nothing stored this renders nothing at all.
    _display_transcription()
    mock_st.text_area.assert_not_called()
    mock_st.container.assert_not_called()
    mock_st.caption.assert_not_called()


def test_display_transcription_shows_transcript(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        "Hello world",
        height=TRANSCRIPT_HEIGHT,
        label_visibility="collapsed",
        key="transcript_b0_0",
    )
    mock_st.subheader.assert_called_once_with(RESULT_HEADING.format("interview.mp3"))


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
        width=BUTTON_WIDTH,
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
        width=BUTTON_WIDTH,
    )


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
        width=BUTTON_WIDTH,
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
    mock_st.subheader.assert_any_call(RESULT_HEADING.format("first.mp3"))
    mock_st.subheader.assert_any_call(RESULT_HEADING.format("second.mp3"))


def test_display_transcription_escapes_filename_in_subheader(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="my_song [live].mp3")]

    _display_transcription()

    mock_st.subheader.assert_called_once_with(RESULT_HEADING.format(r"my\_song \[live\].mp3"))


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

    mock_st.subheader.assert_called_once_with(RESULT_HEADING.format("clip # Big > quote.mp3"))


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
        # `.mp3`), so it must get that type -- Path.suffix would read it as
        # extensionless and serve it as WAV.
        (".mp3", "audio/mpeg"),
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


def _app():
    # The one place the script is loaded for AppTest. default_timeout=5 is a
    # *lowering* from the 30 the file started with (8fca5b9, "faster failure
    # feedback"), not a raise from AppTest's default of 3 for a cold import: the
    # heavy imports are warm from collection, so a run takes ~0.07 s here and
    # the 5 s is headroom for a hung script, not a need.
    return AppTest.from_file(str(APP_PATH), default_timeout=5)


def _run(target):
    """Run `target` and fail if the script crashed, returning the AppTest root.

    AppTest.run() does not raise on an uncaught script exception: it lands in
    at.exception, and `at.error == []` or an empty download list both hold on
    the crashed run, so an absence-only case would pass green on a crash.
    `target` is the AppTest itself or a widget -- set_value() and click() return
    the widget, and Element.run() returns the root. One consequence: a case for
    the st.exception(unexpected) branch cannot go through here, because
    at.exception collects st.exception elements too; it has to build _app() and
    seed the uploader directly.
    """
    at = target.run()
    assert not at.exception, [f"{e.proto.type}: {e.value}" for e in at.exception]
    return at


def _run_app(transcription=None, active_tab=None):
    at = _app()
    if transcription is not None:
        at.session_state["transcription"] = transcription
    if active_tab is not None:
        # AppTest has no tab-selection API; st.tabs' `key` holds the active label.
        at.session_state["input_tabs"] = active_tab
    return _run(at)


def test_page_config():
    assert PAGE_CONFIG == {
        "page_title": "Whisper Transcribe",
        "page_icon": ":material/graphic_eq:",
        "layout": "wide",
    }


def test_app_renders_without_exception():
    at = _run_app()
    assert [t.value for t in at.title] == ["Whisper Transcribe"]


def test_logo_is_the_page_icon_glyph():
    # Not in AppTest's element tree -- the logo ForwardMsg carries no delta and
    # parse_tree_from_messages drops it -- so the call is observed the way
    # _recording observes st.audio_input: patched on the streamlit module the
    # script reaches at call time. Deleting the st.logo line fails only here.
    with patch("streamlit.logo") as logo:
        _run_app()
    logo.assert_called_once_with(":material/graphic_eq:", size="small")


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
    # The button element does not expose `type`, but its proto does. Its
    # width=BUTTON_WIDTH is the one thing about it nothing can pin: width lives
    # on the outer Element proto, which AppTest's Button node discards, and the
    # call is module-level, outside mock_st's reach.
    assert button.proto.type == "primary"


def test_invalid_time_range_shows_inline_error():
    at = _run_app()
    _run(next(t for t in at.text_input if t.label == "Time range").set_value("90,30"))
    assert [e.value for e in at.error] == [_validate_time_range("90,30")]


def test_valid_time_range_shows_no_error():
    at = _run_app()
    _run(next(t for t in at.text_input if t.label == "Time range").set_value("30,90"))
    assert at.error == []


def test_results_render_download_button_with_icon():
    # Seeded results render through the st.fragment(_display_transcription)() wrap.
    at = _run_app([_make_transcription()])
    # at.main, not at: the sidebar's "Settings" heading is a subheader too.
    assert [s.value for s in at.main.subheader] == [RESULT_HEADING.format("interview.mp3")]
    assert at.text_area[0].value == "Hello world"
    download = at.get("download_button")[0]
    assert download.label == "Download"
    assert download.icon == ":material/download:"


def test_no_results_renders_no_download_button():
    assert _run_app().get("download_button") == []


def test_settings_live_in_the_sidebar():
    # Every setting is a sidebar widget and nothing that supplies or acts on
    # audio is. at.sidebar / at.main are scoped views of the same tree, so a
    # control drifting back into the main area fails here and nowhere else --
    # the unscoped accessors the other cases use find a widget wherever it is.
    at = _run_app()
    sidebar, main = at.sidebar, at.main
    assert [s.label for s in sidebar.selectbox] == ["Primary language"]
    assert [t.label for t in sidebar.toggle] == [
        "Translate to English",
        "No verbatim",
        "Decode independently",
    ]
    assert [s.label for s in sidebar.segmented_control] == ["Transcript format"]
    # Both accessors, because AppTest files every expandable block that carries
    # an icon under Status (element_tree.py, `if block.expandable.icon`) and the
    # Advanced options expander has one -- so today it is sidebar.status[0] and
    # at.expander is empty. Reading both keeps the assertion about the widget,
    # not about that classification, should upstream ever correct it.
    assert [e.label for e in (*sidebar.status, *sidebar.expander)] == ["Advanced options"]
    assert [t.label for t in sidebar.text_input] == ["Time range"]
    assert [m.label for m in sidebar.multiselect] == ["Keyterms"]
    assert [b.label for b in sidebar.button] == []
    assert sidebar.tabs == []
    assert [b.label for b in main.button] == ["Transcribe"]
    assert len(main.tabs) == 2
    for widgets in (main.selectbox, main.toggle, main.segmented_control, main.text_input):
        assert widgets == []


def test_main_area_splits_input_and_results_into_two_columns():
    at = _run_app([_make_transcription()])
    input_col, results_col = at.main.columns
    # Equal halves: the input column is what carries the dropzone hint's width
    # and the results column the transcript's measure (see the source comment).
    assert [c.weight for c in at.main.columns] == [0.5, 0.5]
    assert len(input_col.tabs) == 2
    assert [b.label for b in input_col.button] == ["Transcribe"]
    # .get(), not .download_button: the accessor is newer than 1.58.0, and the
    # suite's passing there is a recorded property (CLAUDE.md, Dependencies).
    assert [d.label for d in results_col.get("download_button")] == ["Download"]
    assert input_col.get("download_button") == []
    assert results_col.button == []


def _holder_of(block, label):
    # The first Block, in document order, whose direct children include the
    # element labelled `label` -- i.e. the container a widget was declared in.
    # `block` itself is never a candidate, so a widget declared bare in the
    # column resolves to None (and fails the caller's `is not None`).
    for child in block.children.values():
        if isinstance(child, Block):
            if any(
                not isinstance(g, Block) and getattr(g, "label", None) == label
                for g in child.children.values()
            ):
                return child
            if (found := _holder_of(child, label)) is not None:
                return found
    return None


def test_transcribe_button_is_right_aligned():
    # The Download container's alignment is pinned under mock_st; the Transcribe
    # one is module-level, so only the block proto can pin it. "right" is
    # JUSTIFY_END; "distribute" would be SPACE_BETWEEN, which left-aligns a
    # standalone child and silently unsticks the edge.
    input_col, _ = _run_app().main.columns
    holder = _holder_of(input_col, "Transcribe")
    assert holder is not None and holder.proto.HasField("flex_container")
    flex = holder.proto.flex_container
    assert flex.direction == BlockProto.FlexContainer.Direction.HORIZONTAL
    assert flex.justify == BlockProto.FlexContainer.Justify.JUSTIFY_END


def test_sidebar_is_three_tiers_separated_by_two_seams():
    # The two st.space("small") calls *are* the grouping (input | output |
    # advanced); as source order alone the tiers are invisible. AppTest keeps
    # st.space as an UnknownElement whose .type is "space", so the seams and the
    # heading are readable in order, if not their size.
    sidebar = _run_app().sidebar
    rows = [sidebar.children[i] for i in sorted(sidebar.children)]

    def _name(row):
        if isinstance(row, UnknownElement):
            return row.type
        return getattr(row, "label", None) or getattr(row, "value", None)

    def _kind(row):
        # AppTest files an icon-carrying expander as Status; read it as the
        # expander it is, so an upstream fix to that filing changes nothing here.
        return "Expander" if type(row).__name__ in ("Status", "Expander") else type(row).__name__

    assert [(_kind(r), _name(r)) for r in rows] == [
        ("Subheader", "Settings"),
        ("Selectbox", "Primary language"),
        ("UnknownElement", "space"),
        ("Toggle", "Translate to English"),
        ("ButtonGroup", "Transcript format"),
        ("Toggle", "No verbatim"),
        ("UnknownElement", "space"),
        ("Expander", "Advanced options"),
    ]


def test_every_setting_carries_help():
    # The README restates every tooltip by hand; a widget that loses its help=
    # would otherwise ship silently. The label is the widget's own again, so the
    # tooltip hangs off the widget, not a markdown label beside it.
    sidebar = _run_app().sidebar
    widgets = [
        *sidebar.selectbox,
        *sidebar.toggle,
        *sidebar.segmented_control,
        *sidebar.text_input,
        *sidebar.multiselect,
    ]
    assert len(widgets) == 7
    assert all(w.help for w in widgets), [w.label for w in widgets if not w.help]


def test_time_range_error_renders_beside_the_transcribe_button():
    # The input is in the sidebar's collapsed expander; the alert it produces
    # belongs next to the button it disables, not next to the input.
    at = _run_app()
    _run(next(t for t in at.text_input if t.label == "Time range").set_value("90,30"))
    input_col, results_col = at.main.columns
    assert [e.value for e in input_col.error] == [_validate_time_range("90,30")]
    assert at.sidebar.error == [] and results_col.error == []


def test_empty_results_column_shows_the_hint_until_a_result_exists():
    _, empty = _run_app().main.columns
    assert [c.value for c in empty.caption] == [EMPTY_RESULTS_HINT]
    _, filled = _run_app([_make_transcription()]).main.columns
    assert filled.caption == []


def _publish(at, transcription, batch):
    # Mirrors _handle_transcription's publish: the results plus the batch id
    # that namespaces the transcript widget keys.
    at.session_state["transcription"] = transcription
    at.session_state["batch_id"] = batch
    return _run(at)


def test_new_batch_replaces_previous_transcript_text():
    # A keyed st.text_area restores its session-state value and ignores the
    # `value` argument, so a batch-invariant key renders the *previous* batch's
    # text under the new filename -- and the Download button, whose payload is
    # the text area's return value, serves it. This needs two renders with
    # different data; no single-render test can see it.
    at = _app()
    _publish(at, [_make_transcription(filename="first.mp3")], batch=1)
    assert at.text_area[0].value == "Hello world"

    # Edits must still stick *within* a batch -- that is the point of the key.
    _run(at.text_area[0].set_value("edited by hand"))
    assert at.text_area[0].value == "edited by hand"

    _publish(at, [_make_transcription(filename="second.mp3", text="Second file text")], batch=2)
    assert [s.value for s in at.main.subheader] == [RESULT_HEADING.format("second.mp3")]
    assert at.text_area[0].value == "Second file text"


def test_new_batch_replaces_previous_transcript_when_subtitles_toggled():
    # Flipping include_subtitles changes only the *download* key (txt -> srt),
    # so without the batch namespace the transcript text area stays stale and
    # the SRT cues never reach the screen.
    at = _app()
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
    return _run(at)


def _recording(name="recording.wav", data=b"wav bytes"):
    # A recording for AppTest, planted by patching `streamlit.audio_input` around
    # the run -- the same module-level route the suite already takes for
    # `mlx_whisper.transcribe`, and the only one that works (see _upload). The
    # script does `st.audio_input(...)` at call time, so the patched attribute is
    # what it reaches; no widget is registered, which nothing here depends on.
    # side_effect, not return_value: the real widget hands the script a fresh
    # deepcopy on every run (register_widget in session_state.py), so a shared
    # object whose cursor a first click left at EOF would be a state the app
    # never sees.
    def fresh(*_args, **_kwargs):
        return UploadedFile(UploadedFileRec("id", name, "audio/wav", data), FileURLs())

    return patch("streamlit.audio_input", side_effect=fresh)


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
    previews = _tab(at, UPLOAD_TAB).get("audio")
    assert len(previews) == 1
    # Pins the format= wiring: test_media_mime covers *which* mimetype is chosen,
    # but only a rendered element shows it was passed at all. Dropping format=
    # from the st.audio call serves every preview as .wav again, and nothing
    # else in the suite can see that.
    _assert_declared_mime(previews[0], "audio/mpeg")


def test_upload_preview_is_not_rendered_while_the_record_tab_is_open():
    # The preview loop is gated on `upload_tab.open`. Every st.audio call copies
    # the upload's bytes into the MediaFileManager -- a real second buffer, up
    # to 500 MB per file, since the widget hands the script a deepcopy on each
    # run -- and ungated it was re-emitted on every run for the rest of the
    # session, the Record tab included. The gate is live under AppTest (the
    # default-tab case above renders one element; this renders none), and the
    # suite is green with or without it otherwise, so this is its only pin.
    at = _upload(_run_app(active_tab=RECORD_TAB))
    assert _tab(at, UPLOAD_TAB).get("audio") == []
    # The upload itself is still loaded -- only the preview is gated -- so
    # Transcribe stays enabled through the fallback dispatch.
    assert at.button[0].disabled is False


def test_upload_enables_transcribe():
    at = _upload(_run_app())
    assert at.button[0].disabled is False


def test_transcription_failure_renders_an_escaped_alert():
    # The one case that pushes an _error message through the *real* st.error
    # rather than mock_st: a per-file RuntimeError must surface as an alert on
    # the page, escaped, with no traceback. _run's `assert not at.exception` is
    # the discriminating half here -- the unexpected-exception branch attaches
    # one via st.exception, so a message-only check would not separate the two.
    with patch("mlx_whisper.transcribe", side_effect=RuntimeError("Failed to load audio")):
        at = _upload(_run_app(), name="my_clip.mp3")
        _run(at.button[0].click())

    input_col, results_col = at.main.columns
    # Scoped to the results column: the replay renders where the status does.
    assert [e.value for e in results_col.error] == [
        r"Transcription failed for my\_clip.mp3\: Failed to load audio"
    ]
    # ...and *after* the status block, not inside it. Block.__iter__ yields every
    # descendant, so a column's .error includes alerts rendered into its
    # st.status body -- the original defect, an alert in a collapsed box under a
    # green check, passes the assertion above. Only the status's own scope can
    # tell the two apart; the mocked __exit__-ordering case is the other pin.
    assert results_col.status[0].error == []
    assert input_col.error == [] and at.sidebar.error == []
    assert at.session_state["transcription"] == []
    # No "Transcripts appear here" under the failure on the click run...
    assert results_col.caption == []
    # ...and the hint is back on the next run, when the alerts are gone; the
    # button's click does not survive a rerun, so batch_just_ran is False.
    _run(at)
    _, results_col = at.main.columns
    assert results_col.error == [] and results_col.status == []
    assert [c.value for c in results_col.caption] == [EMPTY_RESULTS_HINT]


def test_result_cards_keep_their_slot_across_the_click_run():
    # The results column reserves a slot above the cards, so the fragment block
    # holding them is the column's second child on the click run (status in the
    # slot) *and* on the run after it (slot empty). Without the slot the cards
    # were index 1 then index 0, and the frontend, which matches blocks by
    # position, remounted every card on the first rerun after a batch.
    def _shape(at):
        _, results_col = at.main.columns
        return [
            [type(g).__name__ for g in child.children.values()]
            for _, child in sorted(results_col.children.items())
            if isinstance(child, Block)
        ]

    with patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app())
        _run(at.button[0].click())
    assert _shape(at) == [["Status"], ["Block"]]
    _run(at)
    assert _shape(at) == [[], ["Block"]]
    # And with no result at all the slot holds the hint's bordered box and
    # nothing follows it -- the hint half of the rule; rendered outside the
    # slot it reads [[], ["Caption"]].
    assert _shape(_run_app()) == [["Block"]]


def test_transcription_status_renders_in_the_results_column():
    with patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app())
        _run(at.button[0].click())
    input_col, results_col = at.main.columns
    assert [(s.label, s.state) for s in results_col.status] == [
        ("Transcribed 1/1 file", "complete")
    ]
    assert input_col.status == []


@pytest.mark.parametrize(
    "tab_sources,expected",
    [
        ([(False, ["upload"]), (True, ["recording"])], ["recording"]),
        ([(False, ["upload"]), (True, [])], ["upload"]),
        ([(False, ["upload"]), (False, ["recording"])], ["upload"]),
        ([(None, ["upload"]), (True, ["recording"])], ["recording"]),
        ([(True, []), (False, [])], []),
    ],
    ids=[
        "open_tab_wins",
        "empty_open_tab_falls_back",
        "no_open_tab_falls_back_in_priority_order",
        "none_reads_as_closed",
        "nothing_loaded",
    ],
)
def test_active_sources(tab_sources, expected):
    # `open_tab_wins` is the only case that fails when the `is_open and` filter
    # is dropped. `none_reads_as_closed` pins the None that TabContainer.open
    # returns without on_change="rerun": a *mixed* layout, because with every
    # flag None the "closed" and "open" readings both yield the priority order
    # and the case would pin nothing -- here None-as-open would return the
    # upload. (Mixed flags never occur in practice; this pins the documented
    # contract, not a reachable state.) The script-level check that the body
    # routes through this helper at all is
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
        _run(at.button[0].click())

    assert [d["filename"] for d in at.session_state["transcription"]] == ["recording.wav"]


def test_transcribe_falls_back_to_a_loaded_upload_from_an_empty_tab():
    # Record tab open with nothing recorded, an upload still loaded: the fallback
    # in _active_sources runs the upload rather than leaving Transcribe dead. This
    # pins the fallback *only* -- it passes with or without the "open tab wins"
    # filter, by design, so a later "only the open tab counts" simplification
    # cannot silently make Transcribe dead. The winning half is the case above.
    with patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT):
        at = _upload(_run_app(active_tab=RECORD_TAB))
        _run(at.button[0].click())

    assert [d["filename"] for d in at.session_state["transcription"]] == ["upload.mp3"]


def test_transcript_format_defaults_to_plain_text():
    at = _run_app()
    assert at.segmented_control[0].value == FORMAT_PLAIN_TEXT


def test_transcript_format_is_required():
    # `required=True` stops a user deselecting the chosen option in the browser
    # ("clicking an already-selected option does nothing"); without it a
    # single-select group returns None, which reads as plain text through the
    # `== FORMAT_SUBTITLES` comparison while looking like nothing is selected.
    # That *behaviour* is invisible to AppTest -- ButtonGroup.unselect() is a
    # no-op for a single-select group either way, verified by deleting the kwarg
    # and re-running -- but the *flag* is not: Element.__getattr__ falls through
    # to the proto, and st.segmented_control(required=True) sets
    # ButtonGroup.required. An earlier version of this file called the kwarg
    # untestable on the strength of the first half alone.
    assert _run_app().sidebar.segmented_control[0].required is True


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
        _run(at.segmented_control[0].set_value(choice))
        _run(at.button[0].click())

    assert at.session_state["transcription"][0]["include_subtitles"] is expected
