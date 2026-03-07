# Automatic Speech Recognition (ASR) Pipeline

Transcribe audio files using the MLX Whisper large-v3-turbo model on Apple Silicon.

## Features

- **Whisper large-v3-turbo** — fast, high-quality transcription via [mlx-whisper](https://pypi.org/project/mlx-whisper/)
- **Apple Silicon acceleration** — native MLX framework (M1/M2/M3/M4)
- **Detailed analysis** — segment-level metrics and word-level timestamps with probabilities
- **Metrics** — audio duration, word count, transcription time
- **Export** — download transcript as plain text or JSON with full segment detail

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

Upload or record an audio file (wav, mp3, m4a, ogg, flac, webm, aac) and click Transcribe.

Results appear in two tabs:

- **Transcript** — full text, copyable
- **Detailed Analysis** — interactive segment table with per-word detail on row selection

## JSON Export

```json
{
  "audio_duration": 10.05,
  "transcript": "transcribed text...",
  "num_words": 42,
  "eval_duration": 3.27,
  "segments": [
    {
      "index": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "transcribed text...",
      "temperature": 0.0,
      "avg_logprob": -0.25,
      "compression_ratio": 1.6,
      "no_speech_prob": 0.01,
      "words": [
        {"word": "transcribed", "start": 0.0, "end": 0.8, "probability": 0.97}
      ]
    }
  ]
}
```

Duration values are in seconds. `eval_duration` is rounded to 2 decimal places. `audio_duration` is `null` if ffprobe cannot determine it.
