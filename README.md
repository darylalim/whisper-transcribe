# Whisper Pipeline

Transcribe or translate audio and video files using the OpenAI Whisper large-v3-turbo model on Apple Silicon with MLX.

## Features

- **Whisper large-v3-turbo** via [mlx-whisper](https://pypi.org/project/mlx-whisper/), accelerated on Apple Silicon
- **99-language transcription** with auto-detect or manual selection
- **Translate non-English audio to English**
- **Multi-file upload** (up to 500 MB per file), **in-browser recording**, **YouTube URL** input via `yt-dlp`, and **audio/video file URL** download via `urllib`
- **Editable subtitle preview** with `.srt` export
- **No-verbatim mode** that removes filler words, false starts, and repetitions
- **Independent-segment decoding** for noisy audio (each 30 s window decodes without context from earlier windows)
- **Time-range clipping** for partial transcription (comma-separated `start,end` pairs in seconds)
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

Upload one or more files (`mp3, m4a, wav, flac, ogg, aac, mp4, mov, webm, mkv`), record audio in-browser, paste a YouTube URL, or paste an audio/video file URL, then click **Transcribe**.

Optional controls:

- **Primary language** — auto-detected by default
- **Translate to English** — translate non-English audio
- **Include subtitles** — initialize an editable SRT preview; the **Download** button serves a `.srt` file instead of `.txt`
- **No verbatim** — remove filler words, false starts, and repetitions
- **Decode segments independently** — disable prior-window context; more robust on noisy or music-heavy audio at the cost of cross-boundary fluency
- **Time range** — transcribe only selected portions; comma-separated `start,end` pairs in seconds (e.g., `30,90` for one clip, `0,60,120,180` for multiple)
- **Keyterms** — bias decoding toward specific terms (proper nouns, jargon)
