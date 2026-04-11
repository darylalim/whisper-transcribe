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

- **Record** / **Upload** tabs (`st.tabs`) — each with audio preview (`st.audio`) and a "Transcribe" button
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results
- `_handle_transcription` reads uploaded file bytes and stores result in `st.session_state`
- `_display_transcription` renders transcript in a read-only text area with a download button

### Audio Formats

aac, flac, m4a, mov, mp3, mp4, ogg, wav, webm

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Testing

- `_transcribe` — mocked `mlx_whisper`, parameter verification
- `_handle_transcription` — session state storage, error handling, bytes/suffix passing
- `_display_transcription` — text area rendering, download button

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)
