# Automatic Speech Recognition (ASR) Pipeline
Transcribe audio files to Markdown text using the MLX Whisper models on Apple Silicon devices.

## Installation
- Install [Python](https://www.python.org/downloads/)
- Install [FFmpeg](https://ffmpeg.org/). Use [Homebrew](https://brew.sh/) on macOS.

Run the following commands in the terminal.

- Set up a Python virtual environment: `python3.12 -m venv streamlit_env`
- Activate the virtual environment: `source streamlit_env/bin/activate` (Mac)
- Install the required Python packages: `pip install -r requirements.txt`
- Run the application in a web browser: `streamlit run streamlit_app.py`

## Features
- Upload .wav or .mp3 audio files
- Select from six Whisper model sizes: tiny, base, small, medium, large, turbo
- Automatic MLX acceleration on Apple Silicon devices
- Export transcript to Markdown text

## Notes
- Only .wav and .mp3 audio files are accepted.
- The default model is `turbo`.
