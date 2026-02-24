# CLAUDE.md

Streamlit web app for automatic speech recognition using the MLX Whisper turbo model on Apple Silicon.

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

- `docling[asr]` — ASR pipeline and MLX Whisper turbo model
- `streamlit` — web UI
- `ffmpeg` — audio processing (system dependency)
- `ruff` — linting and formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Architecture

- `streamlit_app.py` — single-file app entry point
- `tests/test_app.py` — unit tests
- `tests/data/audio/sample_10s.mp3` — sample audio fixture

### Model

Single Docling MLX Whisper turbo variant via `asr_model_specs.WHISPER_TURBO_MLX`, accelerated with `AcceleratorDevice.MPS`.

### Performance

- `@st.cache_resource` on `_get_converter()` — caches `DocumentConverter`
- `ARTIFACTS_PATH` — pre-computed at module level
- `time.perf_counter()` — fractional-second timing

### Input Modes

- **Record** / **Upload** tabs (`st.tabs`) — each with audio preview (`st.audio`) and a "Transcribe" button
- Both paths use `_handle_transcription` which shows inline metrics caption, copyable transcript (`st.code`), and two download buttons (plain text + JSON)

### Audio Formats

wav, mp3, m4a, ogg, flac, webm, aac

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`
- ffprobe failure is non-blocking (duration shows as N/A)
- ffprobe uses plain text output mode (`-show_entries format=duration`)

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `audio_duration` (float | null) — seconds
- `transcript` (string) — generated text
- `num_words` (int) — word count
- `eval_duration` (float) — transcription time in seconds, rounded to 2 decimal places

### Testing

- `_get_audio_duration` — real ffprobe calls and mocked subprocess
- `_transcribe` — mocked `_get_converter` and `ConversionStatus`
- `_handle_transcription` — mocked `st`, `_transcribe`, and `_get_audio_duration`; covers caption formatting, error handling, and temp directory cleanup

## Resources

- [ASR Pipeline with Whisper](https://docling-project.github.io/docling/examples/minimal_asr_pipeline/)
- [ASR Pipeline performance comparison](https://docling-project.github.io/docling/examples/asr_pipeline_performance_comparison/)
- [MLX Whisper example](https://docling-project.github.io/docling/examples/mlx_whisper_example/)
