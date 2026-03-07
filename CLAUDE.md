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

- `docling[asr]` — ASR pipeline; provides `mlx-whisper` as a transitive dependency
- `mlx-whisper` (via docling[asr]) — direct MLX Whisper API for segment-level data
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

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-turbo"`, accelerated with `AcceleratorDevice.MPS`. Called directly (not via Docling) to access raw segment/word-level metrics.

### Performance

- `mlx_whisper.ModelHolder` — caches loaded model in memory across calls
- `time.perf_counter()` — fractional-second timing

### Input Modes

- **Record** / **Upload** tabs (`st.tabs`) — each with audio preview (`st.audio`) and a "Transcribe" button
- Both paths use `_handle_transcription` which stores result in `st.session_state`, and `_display_transcription` renders inner tabs (Transcript / Detailed Analysis), metrics caption, download buttons

### Audio Formats

wav, mp3, m4a, ogg, flac, webm, aac

### Detailed Analysis

- `_show_detailed_analysis` renders segment `st.dataframe` with `on_select="rerun"` and `selection_mode="single-row"`
- Segment columns: #, Start, End, Text, Avg Log Prob, No Speech Prob, Compression Ratio, Temperature
- Word detail on row selection: Word, Start, End, Probability
- `_format_timestamp` for MM:SS.s display

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
- `segments` (array) — per-segment data with fields: index, start, end, text, temperature, avg_logprob, compression_ratio, no_speech_prob, and nested words (word, start, end, probability)

### Testing

- `_get_audio_duration` — real ffprobe calls and mocked subprocess
- `_transcribe` — mocked `mlx_whisper`
- `_handle_transcription` — session state storage, error handling, and temp directory cleanup
- `_display_transcription` — caption rendering, tab creation, JSON export with segments
- `_show_detailed_analysis` — dataframe content and row selection

## Resources

- [ASR Pipeline with Whisper](https://docling-project.github.io/docling/examples/minimal_asr_pipeline/)
- [ASR Pipeline performance comparison](https://docling-project.github.io/docling/examples/asr_pipeline_performance_comparison/)
- [MLX Whisper example](https://docling-project.github.io/docling/examples/mlx_whisper_example/)
