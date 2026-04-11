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
- `tests/data/audio/sample_10s.mp3` — sample audio fixture

### Model

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"`. MLX accelerates natively on Apple Silicon.

### Performance

- `mlx_whisper.ModelHolder` — caches loaded model in memory across calls
- `time.perf_counter()` — fractional-second timing

### Input Modes

- **Upload** / **Record** tabs (`st.tabs`) — each with audio preview (`st.audio`) and a "Transcribe" button
- `_handle_transcription` transcribes and stores result in `st.session_state`
- `_display_transcription` renders output: metrics caption, inner tabs (Transcript / Detailed Analysis), download buttons

### Audio Formats

wav, mp3, m4a, ogg, flac, webm, aac

### Detailed Analysis

- `_show_detailed_analysis` renders segment `st.dataframe` with `on_select="rerun"` and `selection_mode="single-row"`
- Segment columns: #, Start, End, Text, Avg Log Prob, No Speech Prob, Compression Ratio, Temperature
- Word detail on row selection: Word, Start, End, Probability
- `_format_timestamp` formats seconds as MM:SS.s

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
- `segments` (array) — per-segment detail:
  - `index`, `start`, `end`, `text`, `temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob`
  - `words` (array) — per-word detail: `word`, `start`, `end`, `probability`

### Testing

- `_get_audio_duration` — real ffprobe calls and mocked subprocess
- `_transcribe` — mocked `mlx_whisper`, parameter verification
- `_handle_transcription` — session state storage, error handling, temp directory cleanup
- `_display_transcription` — caption, tabs, download buttons, JSON with segments, empty segments fallback
- `_show_detailed_analysis` — dataframe columns/content, timestamp formatting, row selection, empty words edge case

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)
