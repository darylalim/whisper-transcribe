# Whisper Pipeline

Transcribe or translate audio files using the Whisper large-v3-turbo model on Apple Silicon.

## Features

- **Whisper large-v3-turbo** — fast, high-quality transcription via [mlx-whisper](https://pypi.org/project/mlx-whisper/)
- **Apple Silicon acceleration** — native MLX framework (M1/M2/M3/M4)
- **Multilingual** — 99 languages with auto-detect or manual selection
- **Translate to English** — non-English audio → English text
- **Timestamps** — optional sentence- or word-level granularity
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

Upload or record an audio file (aac, flac, m4a, mov, mp3, mp4, ogg, wav, webm) and click **Transcribe**. Optional controls: **Primary language** (auto-detect by default), **Translate to English**, and **Include timestamps** (Off / Sentence / Word).
