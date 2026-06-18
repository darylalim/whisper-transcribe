# CLAUDE.md

Streamlit web app for automatic speech recognition using the OpenAI Whisper large-v3-turbo model on Apple Silicon with MLX.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- Import sorting via ruff with combine-as-imports

## Dependencies

- `mlx-whisper` — speech recognition on Apple Silicon
- `streamlit` — web UI
- `yt-dlp` — YouTube audio download
- `ffmpeg` — audio/video decoding (system dependency)
- `ruff` — linting and formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Architecture

- `streamlit_app.py` — single-file app entry point
- `tests/test_app.py` — unit tests

### Model

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"`. MLX accelerates natively on Apple Silicon. Results are cached via `@st.cache_data(show_spinner=False, max_entries=20)`; the spinner is disabled because per-file progress is rendered by the `st.status` wrapper in `_handle_transcription`.

### Input Modes

- **Upload** / **Record** / **YouTube** / **URL** tabs (`st.tabs`) — Upload accepts multiple files (`accept_multiple_files=True`) with one `st.audio` preview per file; Record takes a single recording with its own preview; YouTube takes a URL (gated by a `youtube.com` / `youtu.be` regex and stripped of whitespace), downloads the best audio stream via `yt-dlp` (cached, `restrictfilenames=True`, `noplaylist=True`), and shows an `st.audio` preview of the bytes; URL takes any `http(s)` audio/video file URL (gated by an `https?://` regex; YouTube URLs short-circuit to an `st.info` redirecting to the YouTube tab), downloads via `urllib.request.urlopen` with a 60-second timeout and a 500 MB cap (`MAX_URL_DOWNLOAD_BYTES`, cached), derives the filename from the URL path (percent-decoded, fallback `download`), and shows an `st.audio` preview
- Below the tabs, in order: **Primary language** selector, **Translate to English** toggle, **Include subtitles** toggle, **No verbatim** toggle (enables `word_timestamps=True` + `hallucination_silence_threshold=2.0` to skip hallucinations on non-speech audio like music outros), **Decode segments independently** toggle (sets `condition_on_previous_text=False` so each 30 s window decodes without prior-window context — robust on noisy audio at the cost of cross-boundary fluency), **Time range** text input (forwarded as `clip_timestamps`; comma-separated `start,end` pairs in seconds, e.g. `30,90` or `0,60,120,180`; blank → `"0"` for the full file), **Keyterms** chip input (`st.multiselect` with `accept_new_options=True`, max 50 chips, joined with `, ` and forwarded as `initial_prompt`), and a right-aligned **Transcribe** button
- The Transcribe button dispatches in priority order: uploaded files → recording → YouTube audio → URL audio. Each non-upload source is wrapped in a single-element list. YouTube and URL sources share a `_RemoteAudio` adapter exposing `.name` (a safe filename, including extension when available) and `.read()` so they flow through `_handle_transcription` without changes. UI flags are routed through `_transcription_kwargs`, which centralizes the `translate → task` mapping and the `decode_independently → condition_on_previous_text` inversion so a script-level negation can't silently disappear. Subtitles controls both the text area's initial content (SRT-formatted segments when on, plain text when off) and the format the **Download** button serves (`.srt` vs `.txt`); the text area is always editable
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results (`language=None` → Whisper auto-detects)
- `_handle_transcription` wraps the batch in `st.status(...)` (label updates to `Transcribing {name} ({i}/{total})...` per file, transitions to `complete` at the end), transcribes each upload, and stores the resulting list of `{result, file_stem, filename, include_subtitles}` dicts in `st.session_state["transcription"]`. The `file_stem` includes the source extension (e.g., `interview_mp3_transcript`) to disambiguate downloads when two uploads share a stem. Per-file errors are reported inline via `st.error` and don't stop the rest of the batch
- `_display_transcription` renders one stacked section per stored result: `st.subheader(filename)` + an editable text area (plain text or SRT segments per `include_subtitles`) + a right-aligned **Download** button that captures the text area's edited content. The button sits in the right column of an `st.columns([3, 1])` split — same ratio as the Transcribe button — so the two share an edge and width. The label is always `Download` regardless of format; the file extension (`.srt` when subtitles are on, `.txt` otherwise) is set via the filename + MIME type args, and a `help=` tooltip (`"Downloads as .srt when subtitles are enabled, .txt otherwise."`) explains the format switch on hover. Indexed widget keys (`transcript_{i}`, `download_{txt,srt}_{i}`) avoid collisions

### Audio Formats

mp3, m4a, wav, flac, ogg, aac, mp4, mov, webm, mkv

### Upload Limit

- Per-file cap of **500 MB**, set in `.streamlit/config.toml` via `server.maxUploadSize`
- Enforced browser-side by Streamlit; with `accept_multiple_files=True` the cap is per-file, not aggregate
- Server restart required after changing this setting

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Testing

Mocked at the boundary (`mlx_whisper`, `yt_dlp`, `urlopen`, `st`). Shared fixtures (`mock_mlx`, `mock_st`, `mock_uploaded_file`) and helpers (`_make_file`, `_stub_urlopen`, `_stub_ytdlp`, `_make_transcription`, `_expected_transcribe_kwargs`, `_handle_transcription_kwargs`, `_ui_state`) factor the common setup; kwarg-forwarding cases are `@pytest.mark.parametrize`d. An autouse `_clear_caches` fixture clears the `@st.cache_data` wrappers before each test so cached results don't leak between cases.

- `_transcribe` — defaults, kwarg forwarding (language, task, initial_prompt, no_verbatim, condition_on_previous_text, clip_timestamps), temp-file cleanup, empty-text guard; `no_verbatim=True` flips `word_timestamps` and `hallucination_silence_threshold`; `clip_timestamps` defaults to `"0"` (full file) and accepts custom ranges (e.g., `"30,90"`)
- `_handle_transcription` — session-state storage, per-file error handling (RuntimeError + unexpected), kwarg forwarding, multi-file batches, partial-failure scenarios
- `_transcription_kwargs` — UI-flag → `_handle_transcription` kwargs mapping; `translate=True` ↔ `task="translate"`, `decode_independently=True` ↔ `condition_on_previous_text=False`; passthrough of `language`, `include_subtitles`, `initial_prompt`, `no_verbatim`, `clip_timestamps`
- `_display_transcription` — filename subheader, editable text area, right-aligned **Download** button in `st.columns([3, 1])`; label is always `Download`; filename, MIME type, and widget key are derived from `include_subtitles` (`.txt` vs `.srt`); `help=` tooltip preserved in both `.txt` and `.srt` paths; edits to the text area flow through to the download payload; multi-file stacked rendering with indexed keys
- `_RemoteAudio` / `_fetch_youtube_audio` / `_fetch_url_audio` — adapter round-trip; YouTube fetch with mocked `yt_dlp` (bytes + filename, `extract_info` call args, safe-download options `format=bestaudio/best`, `noplaylist`, `restrictfilenames`, `quiet`); URL fetch with mocked `urlopen` (bytes + filename, `timeout=60`, query-string stripping, percent-decoded filename, empty-path fallback to `download`, oversized-response `RuntimeError`)
- Formatting helpers — `_format_language` (`None` → `"Detect"`, title-casing of lowercase codes), `_format_timestamp` (zero, minutes/seconds, hours, comma decimal marker), `_format_srt` (single-segment, multi-segment cue separator, `-->` escaping to `->` to keep SRT structure intact)

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)
