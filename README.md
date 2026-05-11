# Whisper Pipeline

Transcribe or translate audio and video files using the Whisper large-v3-turbo model on Apple Silicon.

## Features

- **Whisper large-v3-turbo** via [mlx-whisper](https://pypi.org/project/mlx-whisper/), accelerated on Apple Silicon
- **99-language transcription** with auto-detect or manual selection
- **Translate non-English audio to English**
- **Multi-file upload** (up to 500 MB per file) and **in-browser recording**
- **Editable subtitle preview** with `.srt` export
- **Keyterm biasing** for proper nouns and jargon (up to 50 terms)
- **Cached transcriptions** via `@st.cache_data`

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4)
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

Upload one or more files (`mp3, m4a, wav, flac, ogg, aac, mp4, mov, webm, mkv`) or record audio in-browser, then click **Transcribe**.

Optional controls:

- **Primary language** — auto-detected by default
- **Translate to English** — translate non-English audio
- **Include subtitles** — initialize an editable SRT preview; the download button switches from `.txt` to `.srt`
- **Keyterms** — bias decoding toward specific terms (proper nouns, jargon)
