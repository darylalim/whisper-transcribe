import math
import re
import tempfile
import textwrap
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

# mlx/core is a compiled extension; ty resolves it through the mlx/core/*.pyi
# stubs the locked mlx wheel ships, so this import carries no suppression and
# mx.clear_cache() is type-checked for real. Not every release ships the stubs
# (0.32.0 did not): on a stub-less lock this line needs a line-scoped
# `ty: ignore[unresolved-import]` again, and ty exits 1 on a stale one, so the
# directive tracks the lock in both directions. That token is spelled without
# its leading `#` on purpose -- a `#`-prefixed copy anywhere in a comment is a
# live directive to ty. History and measurements: CLAUDE.md, "Model".
import mlx.core as mx
import mlx_whisper
import streamlit as st
from mlx_whisper.tokenizer import LANGUAGES
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
# Ordered most-likely-first, and deliberately short: the uploader dropzone lists
# these on one `text-overflow: ellipsis` line, so a long list truncates mid-word.
# See "Accepted formats" in CLAUDE.md before adding to either tuple.
AUDIO_FORMATS = (
    "mp3",
    "m4a",
    "wav",
    "opus",
)
VIDEO_FORMATS = (
    "mp4",
    "mov",
    "webm",
    "mkv",
)
# st.audio's `format` is not a hint. It is passed verbatim through
# _marshall_av_media into MediaFileManager.add() with no sniffing anywhere, and
# becomes both the Content-Type header the media route serves and the extension
# in the /media/<hash>.<ext> URL. Its default is "audio/wav", so without this map
# every preview — an .mp3 upload, an .m4a voice memo — is advertised as WAV.
# Chrome and Firefox sniff the container and play it anyway; browsers that trust
# the declared type can refuse. Deliberately not mimetypes.guess_type: it maps
# .m4a to the non-standard audio/mp4a-latm that browsers do not recognize and
# reads a table that varies by platform. One entry per extension the uploader
# *accepts*, which is not quite the eight it declares: normalize_upload_file_type
# (streamlit/elements/lib/file_uploader_utils.py, TYPE_PAIRS) silently adds
# `.mpeg4` beside `.mp4`, browser- and server-side alike, so a clip.mpeg4 upload
# is legal and needs an entry or it lands on the fallback below.
# test_media_mime_covers_every_upload_format keeps this map equal to that
# normalized list, so a format added to AUDIO_FORMATS/VIDEO_FORMATS (or an alias
# Streamlit pairs in) without an entry here fails a test instead of shipping the
# WAV mis-declaration for that one extension.
MEDIA_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "mp4": "video/mp4",
    "mpeg4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
}
# Fallback for an extension outside the map. With the map mirroring the accept
# list and _media_mime reading the extension the way the uploader does, no
# upload reaches it; it must stay non-empty regardless, because the media route
# does `media_type=mimetype or "text/plain"` and an empty string would serve
# audio as text.
DEFAULT_MEDIA_MIME = "audio/wav"
ERROR_ICON = ":material/error:"
# Cap on an error alert's length, applied head-and-tail rather than as a plain
# truncation. The exception text reaching _error was not written for a UI: a
# corrupt file makes mlx_whisper raise RuntimeError("Failed to load audio: " +
# ffmpeg's entire stderr) — measured at 1740 characters over 14 lines, of which
# roughly 1500 are ffmpeg's version banner and its --enable-* configure flags.
# That cost nothing while per-file errors rendered into a collapsed st.status
# nobody had reason to open; making them visible is what turned it into a wall of
# text, so this is a consequence of that fix rather than a pre-existing wart.
#
# Both ends are kept because the two useful halves sit at opposite ends: the head
# carries "Transcription failed for <name>: Failed to load audio:" (what failed)
# and the tail carries "Invalid data found when processing input" (why). Keeping
# only the first line renders the version banner; keeping only the last drops the
# filename and the operation. 240 was chosen by rendering 240/320/400 against the
# real failure — it is the smallest that still lands both ends.
ERROR_MESSAGE_LIMIT = 240
# Characters per subtitle line. 42 is the long-standing broadcast convention
# (Netflix, BBC and EBU all land on 42 for Latin scripts) and it is a *display*
# constraint, not an SRT one — the format itself imposes no limit, which is why
# unwrapped cues are valid, play fine, and still look wrong over a picture.
# Applies to the Subtitles transcript format only; plain text is untouched.
SUBTITLE_LINE_WIDTH = 42
# The transcript format choice, as the segmented control renders it. This governs
# two things at once -- what the results text area shows (plain text vs timestamped
# SRT cues) and which extension the Download button serves (.txt vs .srt) -- which
# is why it is a two-option format picker rather than the "Include subtitles" toggle
# it used to be: a boolean names one of the two states and leaves the other implied,
# so the .srt consequence was reachable only through the help tooltip. Order is
# display order, and FORMAT_PLAIN_TEXT is the default.
FORMAT_PLAIN_TEXT = "Plain text"
FORMAT_SUBTITLES = "Subtitles"
TRANSCRIPT_FORMATS = (FORMAT_PLAIN_TEXT, FORMAT_SUBTITLES)
LANGUAGE_CODES: list[str | None] = [None] + sorted(LANGUAGES, key=lambda c: LANGUAGES[c])
# Width of the Transcribe and Download buttons. Every other control moved into
# the sidebar, where it takes the native stacked form, so the buttons are the
# only two elements left that share a dimension -- one in each main column, at
# the right edge of its column.
#
# 168 is inherited, not chosen for the wide layout: it was reverse-engineered
# from the st.columns([3, 1]) the buttons once sat in when the app was a single
# `centered` column -- the narrow column of a 704px content box, a
# calc(25% - 16px) flex basis plus half the 16px of leftover, i.e.
# (704 - 2*16) / 4 -- and kept so the buttons render exactly as before. Each
# main column measures 714px at a 1920px viewport (708.5 once a result makes
# the page scroll and Chrome's thin scrollbar takes its 11px), so it is the
# same button in the same amount of room. The two do not share a right edge --
# Download sits inside the results card, whose 15px padding plus 1px border
# insets it by 16px, and Transcribe sits in the bare input column -- as they
# have not since aae209d put Download inside that card; see CLAUDE.md.
BUTTON_WIDTH = 168
# Height of each transcript text area. 300 dates from 7b055f1, when the app was
# one default-layout column with the results below the controls, and carried no
# stated rationale; with the results in their own column beside the input, the
# number is what keeps a one-file result -- status line, heading, text area,
# Download -- inside a 1920x1080 display's 839px browser viewport. Measured, not
# picked: Download's bottom lands at y=773 with 66px to spare, and 400 would
# leave 26px, too little for a download bar or any extra browser chrome.
TRANSCRIPT_HEIGHT = 360
# What the results column shows whenever it has no results -- before the first
# batch, and again on the runs after one in which every file failed -- except on
# the run that just rendered that batch's own failure alerts (see
# _display_transcription).
EMPTY_RESULTS_HINT = ":material/subtitles: Transcripts appear here"
PAGE_CONFIG: dict[str, Any] = {
    "page_title": "Whisper Transcribe",
    "page_icon": ":material/graphic_eq:",
    "layout": "wide",
}


def _active_sources(
    tab_sources: Sequence[tuple[bool | None, Sequence[UploadedFile]]],
) -> Sequence[UploadedFile]:
    """Pick the batch to transcribe: the open tab's sources, else the first non-empty.

    `tab_sources` is one (tab.open, sources) pair per input tab, in priority
    order. `tab.open` is None when st.tabs is not tracking state (no
    on_change="rerun"), which reads as "not open" and so as the fallback.
    Factored out of the script body so the selection rule is unit-testable in
    isolation (test_active_sources); the end-to-end check that the script
    actually routes through it, with a recording planted by patching
    st.audio_input, is test_transcribe_runs_the_open_tab_over_a_loaded_upload.
    """
    return next(
        (sources for is_open, sources in tab_sources if is_open and sources),
        next((sources for _, sources in tab_sources if sources), []),
    )


def _media_mime(filename: str) -> str:
    """Content-Type for an st.audio preview, derived from the filename's extension.

    See MEDIA_MIME_TYPES for why st.audio's "audio/wav" default is not good enough
    and why this is a hand-written map rather than mimetypes.guess_type.

    rpartition rather than Path.suffix, to read the extension the way the
    uploader's enforce_filename_restriction does (an endswith check): a file
    literally named `.mp3` is accepted as an MP3, and Path.suffix would call it
    extensionless and hand it the fallback.
    """
    return MEDIA_MIME_TYPES.get(filename.lower().rpartition(".")[2], DEFAULT_MEDIA_MIME)


def _plural(count: int, noun: str) -> str:
    """`1 file` / `2 files`, so the status label never reads `1 file(s)`.

    Every batch renders this string, and a single file is the common case — which
    is exactly the one the parenthesised form gets wrong.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _format_language(code: str | None) -> str:
    return "Detect" if code is None else LANGUAGES[code].title()


def _format_timestamp(seconds: float, decimal_marker: str = ".") -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def _wrap_cue(text: str, width: int = SUBTITLE_LINE_WIDTH) -> str:
    """Wrap one cue's text to `width`, balancing the lines.

    Whisper emits a segment per utterance with no regard for line length, so a
    cue routinely ran well past the ~42 characters subtitle convention allows —
    valid SRT that players accept and that looks wrong burned into a picture.

    Greedy wrapping alone is not enough: `textwrap` fills each line to the brim
    and can leave a one-word orphan (`41 chars` / `3 chars`), which is *more*
    conspicuous on screen than the long line it replaced. So the width is
    narrowed to the smallest value that still produces the same number of lines,
    which spreads the words evenly across them.

    The rule on both `break_*` flags is "never split a token": one over-long line
    reads better than a word cut in half. `break_long_words=False` alone is not
    enough — `textwrap` also splits on hyphens by default, which turned
    `https://example.com/an-extremely-long-path` into two lines mid-URL. A
    hyphenated token now overflows instead, matching how a long word is treated.

    The leading `" ".join(text.split())` normalises whitespace, which also defuses
    a newline inside a segment — an embedded blank line would otherwise terminate
    the cue early and corrupt every cue after it.
    """
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    lines = textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)
    # Do not add a "never narrow below the longest token" floor here. It looks
    # obviously right and is measurably wrong: over 39k generated cues it changed
    # 74 of them and every change was *less* balanced (`[24,24,22,21,30]` became
    # `[24,24,29,14,30]`), while the over-long-token case it was meant to fix —
    # `Visit` / `<64-char URL>` / `now` — comes out byte-identical either way,
    # because a token wider than `width` lands on its own line at every candidate.
    for candidate in range(math.ceil(len(text) / len(lines)), width):
        balanced = textwrap.wrap(text, candidate, break_long_words=False, break_on_hyphens=False)
        if len(balanced) <= len(lines):
            return "\n".join(balanced)
    return "\n".join(lines)


def _format_srt(result: dict) -> str:
    # Blank segments are skipped, and mlx produces them routinely: "if a segment
    # is instantaneous or does not contain text, clear it" (transcribe.py) sets
    # `text` to "" and then *keeps* the segment in all_segments. Emitting one
    # yields a block with an index, a timing line and no text at all — malformed
    # SRT that strict parsers reject. Reachable on any run, not just under No
    # verbatim: the same branch fires whenever `start == end`.
    #
    # Filtering before `enumerate` is what keeps cue numbers contiguous; skipping
    # inside the loop would leave gaps in the sequence.
    cues = (s for s in result["segments"] if s["text"].strip())
    return "\n".join(
        f"{i}\n"
        f"{_format_timestamp(s['start'], decimal_marker=',')} --> "
        f"{_format_timestamp(s['end'], decimal_marker=',')}\n"
        f"{_wrap_cue(s['text'].replace('-->', '->'))}\n"
        for i, s in enumerate(cues, start=1)
    )


_MARKDOWN_ESCAPE_RE = re.compile(r"([\\`*_~\[\]:$&])")


def _escape_markdown(text: str) -> str:
    """Render `text` literally at any of this app's Markdown sinks.

    Filenames are untrusted: an upload carries whatever name the browser sent,
    and `[`, `*`, `_`, `:` and `&` are all legal characters in one.

    Two mechanisms, because Markdown has two layers:

    *Inline* constructs are backslash-escaped. A filename containing *, _,
    backticks, brackets, or : (emoji/Material-icon directives) — underscored
    names are the everyday case — would otherwise mis-render. `&` is in the class
    because micromark's characterReference is a parse-time construct: without it
    `clip&#58;streamlit&#58;.mp3` decodes to a live `:streamlit:` that the
    frontend's post-parse pass swaps for the logo image, and `Rock &amp; Roll.mp3`
    displays as `Rock & Roll.mp3`. `&` is ASCII punctuation, so `\\&` is a valid
    CommonMark characterEscape.

    *Block* constructs are defused by collapsing whitespace, not by escaping:
    headings, blockquotes, lists, thematic breaks and GFM tables all need a line
    start, so removing newlines removes every one of them at once — a smaller
    change than adding #, >, -, +, | and the ordered-list forms to the class.
    This has to happen here rather than at one call site, because *two* sinks
    render without the frontend's isLabel flag (which is what would otherwise
    auto-escape block markers and strip block elements):

    - `st.error`'s body — AlertElement passes isLabel for the alert title only.
    - `st.subheader` — the heading component does `[first, ...rest] = body.split("\\n")`
      and renders `rest` through a bare `StreamlitMarkdown` with no isLabel and no
      disallowedElements, so everything after the first newline is full Markdown.

    No browser path delivers a newline: browsers percent-encode one in an
    upload's name before sending it, and st.audio_input names its own file. A
    direct PUT to the upload route can, since the route copies the multipart
    filename through unchanged — one more reason the collapse stays, on top of
    both sinks being unguarded and the cost being one join.
    """
    return _MARKDOWN_ESCAPE_RE.sub(r"\\\1", " ".join(text.split()))


def _condense(message: str) -> str:
    """Collapse `message` to one line and cap it at both ends.

    See `ERROR_MESSAGE_LIMIT` for why both ends are kept rather than truncating.
    Anything already inside the limit passes through unchanged — every
    `_validate_time_range` result, and every failure whose exception has a
    human-sized string — so this only fires on the pathological cases.
    """
    message = " ".join(message.split())
    if len(message) <= ERROR_MESSAGE_LIMIT:
        return message
    half = ERROR_MESSAGE_LIMIT // 2
    return f"{message[:half].rstrip()} […] {message[-half:].lstrip()}"


def _error(message: str) -> None:
    """Render an error alert with the app's icon, as literal text.

    The *whole* message is escaped, so callers pass raw text — filenames,
    exception strings — and never escape at the interpolation site. Escaping the
    fixed literals along with them is harmless: `:` becomes `\\:`, which renders
    as a colon.

    Both halves matter. st.error's body is one of the two sinks Streamlit renders
    without the frontend's isLabel flag (see _escape_markdown), and the text
    reaching it is untrusted: _handle_transcription interpolates the upload's raw
    filename into both of its failure messages, and _validate_time_range echoes
    the raw input back. An unescaped `![](…)` in either fires an image request to
    whatever destination the text spells — and a filename needs no `/` to point
    off-site, since `https:host` resolves as `https://host/`. Routing every call
    through here also makes the icon structural instead of a literal repeated at
    every call site, where a new one renders a bare box, and gives `_condense` a
    single place to bound the length, which matters for the same reason: the
    text is a library's, not ours. See `ERROR_MESSAGE_LIMIT`.
    """
    st.error(_escape_markdown(_condense(message)), icon=ERROR_ICON)


def _validate_time_range(raw: str) -> str | None:
    """Return an error message if the time-range string is malformed, else None.

    Valid forms: blank (full file) or comma-separated non-negative seconds where
    each complete start,end pair has end > start (e.g. "30,90" or "0,60,120,180").
    A trailing unpaired value is a start that runs to the end of the file (e.g.
    "30" or "0,60,120"), matching mlx_whisper.transcribe's clip_timestamps.
    """
    if not raw:
        return None
    values: list[float] = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            return "Time range has an empty value (check for a stray comma)."
        try:
            value = float(token)
        except ValueError:
            return f"Invalid time range: {token!r} is not a number."
        # float() happily accepts "nan", "inf" and any overflowing literal such
        # as 1e400, and none of them trip the checks below: nan < 0 is False and
        # every nan comparison is False, so a malformed range used to validate
        # clean, enable Transcribe, and fail inside mlx_whisper instead.
        if not math.isfinite(value):
            return f"Invalid time range: {token!r} is not a finite number."
        if value < 0:
            return "Time range values must be non-negative."
        values.append(value)
    for start, end in zip(values[::2], values[1::2]):
        if end <= start:
            return f"Time range end ({end:g}) must be greater than start ({start:g})."
    for prev, cur in pairwise(values):
        if cur < prev:
            return "Time range values must be in increasing order."
    return None


def _split_clips(clip_timestamps: str) -> list[str]:
    """Split a `clip_timestamps` string into one `start,end` string per clip.

    `"0,60,120,180"` → `["0,60", "120,180"]`. A trailing unpaired start keeps its
    open end (`"0,60,120"` → `["0,60", "120"]`), which mlx_whisper reads as
    running to the end of the file.

    This exists because **mlx_whisper does not honour more than one clip.** Its
    decode loop is `for seek_clip_start, seek_clip_end in seek_clips:` wrapping
    `while seek < seek_clip_end:` — `seek_clip_start` is bound and never read,
    and there is no `seek = seek_clip_start`, so `seek` carries over from the
    previous clip and each later range simply continues from wherever the last
    one stopped. Given `0,60,120,180` it decodes 0–180 straight through,
    including the 60–120 the user excluded. (`clip_idx` at `transcribe.py:247` is
    initialised and never incremented — vestigial from the same loop upstream.)
    A single pair is unaffected, because `seek` is initialised to
    `seek_clips[0][0]`, which is why `30,90` was always correct and only the
    multi-clip tail was broken — and why the bug outlived a green test named
    `multi_clip` that asserted only that the string was forwarded.
    """
    values = [t.strip() for t in clip_timestamps.split(",") if t.strip()]
    return [",".join(values[i : i + 2]) for i in range(0, len(values), 2)] or ["0"]


def _merge_transcriptions(results: list[dict]) -> dict:
    """Concatenate per-clip results into the shape a single call returns.

    Segment timestamps are already absolute — mlx_whisper derives them from
    `seek`, which starts at the clip's own start frame — so segments concatenate
    with no timestamp fix-up.

    `id` *is* renumbered. mlx assigns it per call as
    `enumerate(current_segments, start=len(all_segments))`, so every clip restarts
    from 0 and a plain concatenation carries duplicates — a shape no single call
    ever produces. Nothing here reads `id` (`_format_srt` numbers cues with its
    own `enumerate`), so this is about not handing back a malformed dict rather
    than about a live bug.

    `language` comes from the first clip, and the clips *cannot* disagree: with
    `language=None` mlx detects from `pad_or_trim(mel, N_FRAMES)` — the first 30
    seconds of the **whole file**, not of the clip — so every call scores the same
    window and returns the same answer. An earlier version of this docstring said
    they detect independently and could in principle differ; they do not.
    """
    segments = [segment for r in results for segment in r["segments"]]
    return {
        "text": "".join(r["text"] for r in results),
        "segments": [{**segment, "id": i} for i, segment in enumerate(segments)],
        "language": results[0]["language"],
    }


@st.cache_data(show_spinner=False, max_entries=20)
def _transcribe(
    audio_bytes: bytes,
    suffix: str,
    *,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
    condition_on_previous_text: bool = True,
    clip_timestamps: str = "0",
) -> dict:
    # One mlx_whisper call per clip, because it decodes straight through the gaps
    # between them — see _split_clips. A single clip (the "0" default, and every
    # plain `start,end` pair) takes exactly one call and its result is returned
    # untouched, so the common path is byte-identical to before. The accepted cost
    # on the multi-clip path is that each call re-runs ffmpeg over the whole file;
    # clips are typically two or three, and decoding the ranges the user actually
    # asked for is worth more than one shared decode of ranges they excluded.
    clips = _split_clips(clip_timestamps)
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        results = [
            mlx_whisper.transcribe(
                tmp.name,
                path_or_hf_repo=ASR_MODEL_REPO,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=no_verbatim,
                hallucination_silence_threshold=2.0 if no_verbatim else None,
                clip_timestamps=clip,
            )
            for clip in clips
        ]
    result = results[0] if len(results) == 1 else _merge_transcriptions(results)
    # Only the merged text is checked, so one silent clip inside a multi-clip
    # range does not fail the whole transcription.
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result


def _handle_transcription(
    uploaded_files: Sequence[UploadedFile],
    *,
    language: str | None,
    task: str,
    include_subtitles: bool,
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
    condition_on_previous_text: bool = True,
    clip_timestamps: str = "0",
) -> None:
    transcriptions: list[dict] = []
    # Per-file failures are collected here and rendered *after* the status block,
    # not inside it. st.status collapses on its first update(label=...): update()
    # clears the proto's `expanded` field unless it is passed again (see
    # mutable_status_container.py), and the frontend's label-change branch then
    # resets the open state to that now-false backend value. So an alert written
    # into the status body during the loop landed inside a collapsed container
    # carrying a *success* indicator: a failed file rendered a green check,
    # "Transcribed 0/1 file", and no visible explanation anywhere on the page. The
    # status stays expandable, so the text was reachable — but nothing on screen
    # suggested a failure had happened or that anything was hidden, which is the
    # part that made it a defect rather than a disclosure.
    failures: list[tuple[str, Exception | None]] = []
    # Publish up front so a previous batch is cleared even if nothing succeeds
    # here. Streamlit interrupts a running script at the next ForwardMsg — the
    # status.update() below is such a point, and RerunException is a
    # BaseException that `except Exception` will not catch — so assigning only
    # after the loop would discard every file already transcribed.
    st.session_state["transcription"] = transcriptions
    # Bump the batch id alongside that publish so _display_transcription's widget
    # keys change with the batch. A keyed st.text_area restores its session-state
    # value and ignores the `value` argument, so reusing transcript_{i} across
    # batches renders the *previous* batch's text under the new filename — and the
    # Download button, whose payload is the text area's return value, serves it.
    st.session_state["batch_id"] = st.session_state.get("batch_id", 0) + 1
    total = len(uploaded_files)
    try:
        with st.status(f"Transcribing {_plural(total, 'file')}...", expanded=True) as status:
            for i, uploaded_file in enumerate(uploaded_files, start=1):
                # Escape before interpolating anywhere Markdown renders. An st.status
                # label takes the Markdown label subset — which includes images, so a
                # filename carrying `![](https:host)` (no `/` needed; it resolves to
                # https://host/) would fetch on *every* file, not just a failure —
                # and st.error below renders full Markdown.
                # See _escape_markdown for what it does and does not cover.
                name_md = _escape_markdown(uploaded_file.name)
                status.update(label=f"Transcribing {name_md} ({i}/{total})...")
                name = Path(uploaded_file.name)
                # Rewind before reading. UploadedFile subclasses io.BytesIO, and
                # read() leaves its cursor at EOF. The rewind covers a same-object
                # double read within one run -- st.file_uploader / st.audio_input
                # hand the script a deepcopy of the cached widget value on every
                # run (register_widget in session_state.py), so no rerun path
                # delivers an already-read object. An earlier version of this
                # comment claimed the cached object survived reruns; it does not.
                # st.audio also rewinds as a side effect (_marshall_av_media calls
                # data.seek(0)), but that is a display call, not a contract, and
                # the Record tab renders no preview.
                uploaded_file.seek(0)
                try:
                    result = _transcribe(
                        uploaded_file.read(),
                        name.suffix,
                        language=language,
                        task=task,
                        initial_prompt=initial_prompt,
                        no_verbatim=no_verbatim,
                        condition_on_previous_text=condition_on_previous_text,
                        clip_timestamps=clip_timestamps,
                    )
                    transcriptions.append(
                        {
                            "result": result,
                            "file_stem": f"{name.stem}_{name.suffix.lstrip('.')}_transcript",
                            "filename": uploaded_file.name,
                            "include_subtitles": include_subtitles,
                        }
                    )
                    # Redundant while session_state holds this exact list (append
                    # mutates it in place), but kept explicit so the progressive
                    # publish does not silently break if `transcriptions` is ever
                    # rebound rather than mutated.
                    st.session_state["transcription"] = transcriptions
                except RuntimeError as e:
                    failures.append((f"Transcription failed for {uploaded_file.name}: {e}", None))
                except Exception as e:
                    failures.append((f"Unexpected error for {uploaded_file.name}: {e}", e))
            status.update(
                label=f"Transcribed {len(transcriptions)}/{_plural(total, 'file')}",
                state="error" if failures else "complete",
            )
    finally:
        # Reclaim MLX's allocator cache now the batch is done. mlx keeps freed
        # device buffers on a free list instead of returning them, so a finished
        # transcription leaves them resident for the life of the server process.
        # Measured here against whisper-large-v3-turbo with word_timestamps=True:
        # 894 MB held after an 8-second file and 1.25 GB after two minutes, all of
        # it reclaimed, for 4.2 ms. The next run costs 0.76 s against 0.74 s with
        # the cache warm — it re-allocates rather than reloading, because the
        # *model* lives in `active` memory (a flat 1543 MB across every
        # measurement) and clear_cache() does not touch it.
        #
        # Verify with mx.get_cache_memory(), NOT with `top`. RSS moved 2.6 MB while
        # 894 MB was reclaimed, because these are device-mapped MTLBuffers — anyone
        # checking the process's resident size concludes this did nothing.
        #
        # mx.set_cache_limit() is the obvious cheaper alternative and is worse: it
        # caps the free list *during* decode, throttling the run it is meant to
        # help. A batch boundary costs nothing in the middle of the work.
        #
        # `finally`, not a plain call after the block: a tab switch or any widget
        # raises RerunException mid-batch, and an interrupted run would otherwise
        # hold its buffers until whenever the next transcription finishes.
        mx.clear_cache()
    # Replayed outside the status block, so the alerts land in the page body where
    # they stay visible. A BaseException (RerunException) unwinds past this without
    # replaying anything, which is correct: that run's output is discarded whole.
    for message, unexpected in failures:
        _error(message)
        if unexpected is not None:
            st.exception(unexpected)


def _transcription_kwargs(
    *,
    language: str | None,
    translate: bool,
    include_subtitles: bool,
    initial_prompt: str | None,
    no_verbatim: bool,
    decode_independently: bool,
    clip_timestamps: str,
) -> dict:
    return {
        "language": language,
        "task": "translate" if translate else "transcribe",
        "include_subtitles": include_subtitles,
        "initial_prompt": initial_prompt,
        "no_verbatim": no_verbatim,
        "condition_on_previous_text": not decode_independently,
        "clip_timestamps": clip_timestamps,
    }


def _display_transcription(*, batch_just_ran: bool = False) -> None:
    transcriptions = st.session_state.get("transcription") or []
    if not transcriptions:
        if batch_just_ran:
            # This run's _handle_transcription rendered its own status and
            # failure alerts directly above: every file failed (or a rerun
            # interrupted the batch before its first file). A "nothing has
            # happened here" hint under them would be wrong, so this run renders
            # nothing. On every later run those alerts are gone and the list is
            # still [] -- the pre-loop publish leaves it that way -- so the hint
            # returns rather than leaving the column blank for the rest of the
            # session, which is what keying on the *key's* absence did.
            return
        # Empty state for the results column. In a single column the output could
        # only ever land below the input; beside it, a first-time visitor meets
        # half a screen of nothing with no sign of what fills it. One muted line
        # in a 54px bordered box at the top of the column -- level with the tab
        # strip across the gap, not with the 68px dropzone below it -- in exactly
        # the slot the st.status box takes once a run starts, so the swap is
        # positionally seamless. text_alignment on the caption, not
        # horizontal_alignment on the box: the caption is a width="stretch"
        # element, so centring *it* changes nothing.
        with st.container(border=True):
            st.caption(EMPTY_RESULTS_HINT, text_alignment="center")
        return
    # Namespaces the widget keys below by batch; see _handle_transcription.
    batch = st.session_state.get("batch_id", 0)
    for i, data in enumerate(transcriptions):
        include_subtitles = data["include_subtitles"]
        if include_subtitles:
            initial = _format_srt(data["result"])
        else:
            initial = data["result"]["text"].strip()
        # One bordered box per result. Sections stack flat otherwise, so in a
        # multi-file batch one file's Download button abuts the next file's
        # heading with nothing marking the seam. No-op visually for a single file.
        with st.container(border=True):
            st.subheader(_escape_markdown(data["filename"]))
            transcript = st.text_area(
                "Transcript",
                initial,
                height=TRANSCRIPT_HEIGHT,
                label_visibility="collapsed",
                key=f"transcript_b{batch}_{i}",
            )
            ext, mime = (
                ("srt", "application/x-subrip") if include_subtitles else ("txt", "text/plain")
            )
            with st.container(horizontal=True, horizontal_alignment="right"):
                st.download_button(
                    "Download",
                    transcript,
                    f"{data['file_stem']}.{ext}",
                    mime,
                    icon=":material/download:",
                    key=f"download_{ext}_b{batch}_{i}",
                    # The second sentence is not padding. st.download_button
                    # materializes non-callable `data` at *render* time and hands
                    # the frontend a pre-baked URL, while st.text_area commits
                    # only on blur or Ctrl/Cmd+Enter — so a click with an
                    # uncommitted edit serves the previous text. Verified in real
                    # Chrome: first click got the pre-edit string, second got the
                    # edit. Not fixable in-app (st.form rejects download buttons,
                    # and a deferred callable runs before the pending update
                    # lands), so the tooltip is the mitigation.
                    help=(
                        "Downloads as .srt when subtitles are enabled, .txt otherwise. "
                        "Commit an edit first — click outside the box or press "
                        "Ctrl/Cmd+Enter — or the download will miss it."
                    ),
                    # Nothing here depends on a post-download rerun — the payload
                    # is the text area's already-committed return value.
                    on_click="ignore",
                    width=BUTTON_WIDTH,
                )


# UI
st.set_page_config(**PAGE_CONFIG)
st.title("Whisper Transcribe")
# Orientation for a first-time visitor, who otherwise meets a bare title and a
# dropzone. st.caption rather than st.info: design.md scopes the callout styles to
# instructions and problems, and this is neither. Deliberately says "transcribed"
# rather than a flat "nothing leaves your Mac" — the first transcription fetches
# the model weights, so the stronger claim would be false for exactly the
# first-time visitor this line exists for.
st.caption("Audio and video, transcribed and translated locally on your Mac.")

# Every setting lives in the sidebar. They apply to whichever tab supplies the
# audio, which is exactly what the layout guidance reserves the sidebar for, and
# moving them out of the main area is what frees it for the input | results
# split below. They take the native stacked form (label above, control below at
# the widget's own width) rather than the label-left/control-right rows the main
# column used to carry: the sidebar's content box is 256px at its default width, and the
# "Transcript format" label beside its 173px segmented control measured 314px,
# so that grid wraps onto two lines here. Help text hangs off each widget's own
# `help=` for the same reason -- the label is the widget's again.
with st.sidebar:
    st.subheader("Settings")
    language = st.selectbox(
        "Primary language",
        LANGUAGE_CODES,
        format_func=_format_language,
        help=(
            # Not "an uploaded file": only one of the two input modes is an
            # upload, and this selector governs both.
            "The primary language spoken in the audio. "
            "By default, the primary language will be detected automatically."
        ),
    )

    # The intent seam: language describes the input, the three controls below
    # shape the output. Uniform spacing renders all four as one undifferentiated
    # run, so the grouping is these two st.space calls, not the source order.
    st.space("small")

    translate = st.toggle(
        "Translate to English",
        help="Translates audio to English instead of transcribing in the source language.",
    )
    # required=True on top of default= is what makes the return a `str` rather
    # than `str | None`: without it, clicking the selected segment deselects it
    # and a single-select segmented control returns None, which would silently
    # read as "plain text" here and give the control a third, invisible state.
    transcript_format = st.segmented_control(
        "Transcript format",
        TRANSCRIPT_FORMATS,
        default=FORMAT_PLAIN_TEXT,
        required=True,
        help="Plain text, or timestamped SRT subtitle cues — best for adding "
        "subtitles to a video. The transcript stays editable either way, and "
        "this also switches the Download button between .txt and .srt.",
    )
    include_subtitles = transcript_format == FORMAT_SUBTITLES
    no_verbatim = st.toggle(
        "No verbatim",
        help="Skips silent stretches where Whisper appears to be hallucinating text, "
        "such as over music or applause after speech ends. Does not remove "
        "filler words or repetitions.",
    )
    # The second seam, matching the one above: input | output | advanced. With
    # only the first one the sidebar reads as two tiers rather than three.
    st.space("small")

    with st.expander("Advanced options", icon=":material/tune:"):
        # "Decode independently", not "Decode segments independently": the
        # expander's content box is 222px at the default sidebar width in Chrome
        # (204px under headless Playwright, whose sidebar is 238px wide), and
        # after the 40px switch, the 16px help icon and the 6px gap before it the
        # label gets 160px (142px headless). The longer label needs 191px and
        # wrapped onto two lines in both -- the only wrapping label in the app,
        # with its help icon pushed to the far edge; this one is 132px. What is
        # decoded independently is in the help text.
        decode_independently = st.toggle(
            "Decode independently",
            help="When enabled, each 30-second window is transcribed without context "
            "from prior windows. More robust on noisy or music-heavy audio.",
        )
        time_range_input = st.text_input(
            "Time range",
            placeholder="e.g., 30,90",
            help='Comma-separated start,end pairs in seconds (e.g., "30,90" for a '
            'single clip, "0,60,120,180" for multiple clips). Leave blank to '
            "transcribe the full file.",
        ).strip()
        time_range_error = _validate_time_range(time_range_input)
        clip_timestamps = time_range_input or "0"
        keyterms = st.multiselect(
            "Keyterms",
            options=[],
            accept_new_options=True,
            max_selections=50,
            placeholder="Add keyterms...",
            help="Up to 50 keyterms to be boosted during transcription. "
            "Boosted terms are more likely to appear in the output.",
        )
    initial_prompt = ", ".join(keyterms) or None

# Input on the left, results on the right. Two equal columns rather than one
# stacked wide column: at a 1920px viewport each measures 714px (722 at the 16px
# default gap; 708.5 once a result makes the page scroll in Chrome and its thin
# scrollbar takes 11px), within ~20px of the 704px content box the whole app
# had as a single `centered` column, so the 168px buttons and every alignment
# measured for that layout carry over -- and a transcript wraps at ~110
# characters per line on the 680px text area (measured; 674.5px and still ~110
# once the scrollbar bites, ~138 with the sidebar collapsed) instead of ~235
# across a 1460px content box. The split is also
# what puts the results beside the input: stacked, a transcript started below
# the fold of a 1080p display before the change. gap="medium" (32px) rather than
# the 16px default because that default equals the vertical gap between widgets
# inside a column, which made the Transcribe button read as attached to the
# results card beside it. The cost of the column being fluid is that the
# dropzone hint is no longer viewport-independent: see "Accepted formats" in
# CLAUDE.md for the ~1472px viewport below which it truncates (the slot it fills
# is 357px at 1472 and 356 at 1470, one short of the text; measured).
input_col, results_col = st.columns(2, gap="medium")

with input_col:
    upload_tab, record_tab = st.tabs(
        [
            ":material/upload: Upload",
            ":material/mic: Record",
        ],
        # on_change="rerun" enables the per-tab `.open` flag that the source
        # dispatch below reads; `key` exposes the active tab's label in session
        # state so AppTest can drive tab switches (it has no tab-selection API).
        on_change="rerun",
        key="input_tabs",
    )
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload audio or video files",
            type=AUDIO_FORMATS + VIDEO_FORMATS,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        for uploaded_file in uploaded_files:
            st.audio(uploaded_file, format=_media_mime(uploaded_file.name))

    with record_tab:
        # No st.audio preview here, unlike the Upload tab. st.audio_input is not
        # a bare capture control — it renders its own WaveSurfer player
        # (interactive waveform, timecode, Play/Pause as soon as a recording
        # exists, and a "Clear recording" action), so an st.audio call would
        # stack a second, visually different player on the same bytes.
        recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")

    # The tab the user is looking at wins, and that is not what a flat priority
    # chain does. Upload and Record declare their widgets in ungated tab bodies,
    # so both values are sticky across tab switches — an upload stays loaded
    # while the user records on the other tab. A plain
    # `uploaded_files or [recorded_audio]` chain would let that earlier upload
    # outrank the recording whose player is on screen, and the results
    # subheader, which arrives after a full model pass, would be the only sign.
    #
    # The old order is kept as the *fallback*, for when the open tab has no
    # source of its own — an empty Record tab with an earlier upload still
    # loaded — so the button never goes dead while a usable source exists. That
    # leaves a deliberate residue: in exactly that case Transcribe still runs
    # the upload.
    audio_sources = _active_sources(
        (
            (upload_tab.open, uploaded_files),
            (record_tab.open, [recorded_audio] if recorded_audio else []),
        )
    )
    # Rendered here, directly above the button it disables, rather than in the
    # sidebar under the (collapsed by default) expander that holds the input: a
    # disabled Transcribe always shows its reason in the same column.
    if time_range_error:
        _error(time_range_error)
    with st.container(horizontal=True, horizontal_alignment="right"):
        transcribe_clicked = st.button(
            "Transcribe",
            icon=":material/graphic_eq:",
            type="primary",
            disabled=not audio_sources or bool(time_range_error),
            width=BUTTON_WIDTH,
        )

with results_col:
    # The st.status that _handle_transcription opens renders here, above the
    # results it is reporting on, rather than under the Transcribe button.
    batch_just_ran = bool(transcribe_clicked and audio_sources and not time_range_error)
    if batch_just_ran:
        _handle_transcription(
            audio_sources,
            **_transcription_kwargs(
                language=language,
                translate=translate,
                include_subtitles=include_subtitles,
                initial_prompt=initial_prompt,
                no_verbatim=no_verbatim,
                decode_independently=decode_independently,
                clip_timestamps=clip_timestamps,
            ),
        )

    # Wrapped in a fragment so transcript edits/downloads rerun only this section
    # instead of the whole script (which re-evaluates the sidebar and both input
    # tabs). The flag is a fragment argument: st.fragment stores the call's
    # arguments and replays them on a fragment-only rerun, and it is stale there
    # by design -- a fragment rerun is a text-area edit or a download, neither of
    # which exists while the column is empty.
    st.fragment(_display_transcription)(batch_just_ran=batch_just_ran)
