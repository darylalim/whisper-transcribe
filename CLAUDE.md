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
- `ffmpeg` — audio processing (system dependency)
- `ruff` — linting and formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Architecture

- `streamlit_app.py` — single-file app entry point
- `tests/test_app.py` — unit tests

### Model

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"`. MLX accelerates natively on Apple Silicon. Results are cached via `@st.cache_data`.

### Input Modes

- **Upload** / **Record** tabs (`st.tabs`) — each contains its source widget plus an audio preview (`st.audio`)
- Below the tabs: a **Primary language** selector, a **Translate to English** toggle, an **Include timestamps** dropdown (Off / Sentence / Word), and a single right-aligned **Transcribe** button
- The button dispatches whichever input has content (uploaded file preferred over recording when both are present); the toggle maps to Whisper's `task="translate"` (force English output) vs `task="transcribe"` (output in source language); the timestamps dropdown maps to mlx-whisper's `word_timestamps=True` only when set to "Word", and the display formats segments accordingly
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results (`language=None` → Whisper auto-detects)
- `_handle_transcription` reads uploaded file bytes and stores result in `st.session_state`
- `_display_transcription` renders transcript in a read-only text area with a download button

### Audio Formats

aac, flac, m4a, mov, mp3, mp4, ogg, wav, webm

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Testing

- `_transcribe` — mocked `mlx_whisper`, kwarg verification (language, task, word_timestamps), defaults, temp-file cleanup, empty-text guard
- `_handle_transcription` — session state storage, error handling, argument forwarding (including timestamps mode → `word_timestamps`)
- `_display_transcription` — text area + download button rendering for None / Sentence / Word modes
- Formatting helpers — `_format_timestamp`, `_format_segments_with_timestamps`, `_format_words_with_timestamps`

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)
