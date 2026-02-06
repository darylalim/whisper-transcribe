# CLAUDE.md

## Project Overview

Streamlit web app for automatic speech recognition using MLX Whisper models on Apple Silicon.

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Commands

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Test**: `pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `docling[asr]` — ASR pipeline and MLX Whisper models
- `streamlit` — web UI
- `ffmpeg` — audio processing (system dependency)
- `ruff` — linting/formatting (dev)

## Architecture

### Entry Point

`streamlit_app.py` — single-file app.

### Models

Docling MLX Whisper variants via `asr_model_specs`, accelerated with `AcceleratorDevice.MPS`:

- `WHISPER_TINY_MLX`, `WHISPER_BASE_MLX`, `WHISPER_SMALL_MLX`
- `WHISPER_MEDIUM_MLX`, `WHISPER_LARGE_MLX`, `WHISPER_TURBO_MLX`

### Performance

- `@st.cache_resource` on `_get_converter()` to cache `DocumentConverter` per model
- `time.perf_counter()` for timing (fractional seconds)
- `MODEL_NAMES` pre-computed from `MODEL_OPTIONS.keys()` to avoid repeated list creation

### Audio Formats

Supported: wav, mp3, m4a, ogg, flac, webm, aac

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()` for debugging
- ffprobe failure is non-blocking — transcription proceeds with duration as N/A
- ffprobe uses plain text output mode (`-show_entries format=duration`) instead of JSON

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model name
- `audio_duration` (float | null) — audio duration in seconds
- `transcript` (string) — generated text
- `num_words` (int) — word count
- `eval_duration` (float) — transcription time in seconds (rounded to 2 decimal places)

### Metrics

`st.metric` displays all JSON fields except transcript.

## Resources

- [ASR Pipeline with Whisper](https://docling-project.github.io/docling/examples/minimal_asr_pipeline/)
- [ASR Pipeline performance comparison](https://docling-project.github.io/docling/examples/asr_pipeline_performance_comparison/)
- [MLX Whisper example](https://docling-project.github.io/docling/examples/mlx_whisper_example/)
