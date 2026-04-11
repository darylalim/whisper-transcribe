# Whisper Pipeline

Transcribe audio files using the Whisper large-v3-turbo model on Apple Silicon.

## Features

- **Whisper large-v3-turbo** — fast, high-quality transcription via [mlx-whisper](https://pypi.org/project/mlx-whisper/)
- **Apple Silicon acceleration** — native MLX framework (M1/M2/M3/M4)
- **Cached results** — repeat transcriptions return instantly via `@st.cache_data`
- **Export** — download transcript as plain text

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- [FFmpeg](https://formulae.brew.sh/formula/ffmpeg)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
brew install ffmpeg
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

Record or upload an audio file (aac, flac, m4a, mov, mp3, mp4, ogg, wav, webm) and click Transcribe.
