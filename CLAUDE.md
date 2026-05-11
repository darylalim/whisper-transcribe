# CLAUDE.md

Streamlit web app for automatic speech recognition using the Whisper large-v3-turbo model on Apple Silicon.

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

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- Import sorting via ruff with combine-as-imports

## Dependencies

- `mlx-whisper` — speech recognition on Apple Silicon
- `streamlit` — web UI
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

- **Upload** / **Record** tabs (`st.tabs`) — Upload accepts multiple files (`accept_multiple_files=True`) with one `st.audio` preview per file; Record takes a single recording with its own preview
- Below the tabs, in order: **Primary language** selector, **Translate to English** toggle, **Include subtitles** toggle, **Keyterms** chip input (`st.multiselect` with `accept_new_options=True`, max 50 chips, joined with `, ` and forwarded as `initial_prompt`), and a right-aligned **Transcribe** button
- The Transcribe button dispatches uploaded files when present; otherwise it wraps the recording in a single-element list. Translate maps to `task="translate"` (English output) vs `task="transcribe"` (source language). Subtitles controls both the text area's initial content (SRT-formatted segments when on, plain text when off) and the single download button rendered (`.srt` vs `.txt`); the text area is always editable
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results (`language=None` → Whisper auto-detects)
- `_handle_transcription` wraps the batch in `st.status(...)` (label updates to `Transcribing {name} ({i}/{total})...` per file, transitions to `complete` at the end), transcribes each upload, and stores the resulting list of `{result, file_stem, filename, include_subtitles}` dicts in `st.session_state["transcription"]`. The `file_stem` includes the source extension (e.g., `interview_mp3_transcript`) to disambiguate downloads when two uploads share a stem. Per-file errors are reported inline via `st.error` and don't stop the rest of the batch
- `_display_transcription` renders one stacked section per stored result: `st.subheader(filename)` + an editable text area (plain text or SRT segments per `include_subtitles`) + a single download button (`.txt` or `.srt`) that captures the text area's edited content. Indexed widget keys (`transcript_{i}`, `download_{txt,srt}_{i}`) avoid collisions

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

- `_transcribe` — mocked `mlx_whisper`, kwarg verification (language, task, initial_prompt), defaults, temp-file cleanup, empty-text guard
- `_handle_transcription` — session state storage as a list, per-file error handling (RuntimeError + unexpected), argument forwarding (`include_subtitles`, `initial_prompt`), multi-file batches, partial-failure scenarios
- `_display_transcription` — filename subheader + editable text area + single download button (`.txt` when subtitles off, `.srt` when on); both buttons capture the text area's edited content; multi-file stacked rendering with indexed keys
- Formatting helpers — `_format_timestamp` (with optional comma decimal marker), `_format_srt` (single-segment + multi-segment cue separator + `-->` escaping to `->` to keep the SRT structure intact)

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)
