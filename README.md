# Automatic Speech Recognition (ASR) Pipeline

Transcribe audio files to Markdown using MLX Whisper models on Apple Silicon.

## Features

- **6 Whisper models**: tiny, base, small, medium, large, turbo
- **MPS acceleration**: Optimized for Apple Silicon (M1/M2/M3/M4)
- **Metrics dashboard**: Model, processing time, word count, speed multiplier
- **JSON export**: Download transcript with detailed timing metrics

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- FFmpeg

## Installation

1. Install FFmpeg (via [Homebrew](https://brew.sh/)):
   ```bash
   brew install ffmpeg
   ```

2. Set up and activate virtual environment:
   ```bash
   python3.12 -m venv streamlit_env
   source streamlit_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
streamlit run streamlit_app.py
```

Upload a WAV or MP3 file, select a model, and click Transcribe.

## JSON Export Format

```json
{
  "model": "turbo",
  "response": "transcribed text...",
  "total_duration": 1234567890,
  "load_duration": 123456789,
  "prompt_eval_count": 5,
  "prompt_eval_duration": 30864197,
  "eval_count": 150,
  "eval_duration": 123456789
}
```

All duration values are in nanoseconds.
