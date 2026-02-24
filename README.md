# Automatic Speech Recognition (ASR) Pipeline

Transcribe audio files to Markdown using the MLX Whisper turbo model on Apple Silicon.

## Features

- **Whisper turbo model** — fast, high-quality transcription via Docling
- **Apple Silicon acceleration** — MPS via MLX framework (M1/M2/M3/M4)
- **Converter caching** — cached for fast repeated transcriptions
- **Metrics dashboard** — audio duration, word count, eval duration
- **JSON export** — download transcript with metrics

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

Upload an audio file (wav, mp3, m4a, ogg, flac, webm, aac) and click Transcribe.

## JSON Export

```json
{
  "audio_duration": 10.05,
  "transcript": "transcribed text...",
  "num_words": 42,
  "eval_duration": 3.27
}
```

Duration values are in seconds. `eval_duration` is rounded to 2 decimal places. `audio_duration` is `null` if ffprobe cannot determine it.
