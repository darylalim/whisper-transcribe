# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application that transcribes audio files (WAV, MP3) to Markdown using MLX Whisper models. The pipeline is optimized for Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS) acceleration.

## Directory Structure

The application is a single file Streamlit app (`streamlit_app.py`).

## Main Dependencies

- `ffmpeg` - FFmpeg is a tool for audio duration detection.
- `docling[asr]` - Document conversion framework with automatic speech recognition (ASR) pipeline support.
- `streamlit` - Web user interface framework.

MLX Whisper models are downloaded automatically on first use.

## Architecture

### Components in `streamlit_app.py`

- **transcribe()** - Core function that loads the model, runs ASR, and returns metrics dict
- **Utility Functions** - `format_duration()`, `format_bytes()`, `get_audio_duration()`
- **Streamlit UI** - File upload, model selection, transcription display, metrics dashboard, JSON export
- **Data Flow:** User Upload → Temp File → ASR Pipeline → DoclingDocument → Markdown Export → Display + JSON Download

### Download

Include these items in the response JSON file for download.

- model (string): Model name
- response (string): The model's generated text response
- total_duration (integer): Time spent generating the response in nanoseconds
- load_duration (integer): Time spent loading the model in nanoseconds
- prompt_eval_count (integer): Number of audio segments processed
- prompt_eval_duration (integer): Time spent decoding audio in nanoseconds
- eval_count (integer): Number of words in the transcript
- eval_duration (integer): Time spent transcribing in nanoseconds

Use `time.perf_counter_ns()` to measure duration and return time in nanoseconds.

### Metrics

Display a summary row (Model, Total Time, Words, Speed) followed by an expander with detailed metrics:

- Audio File: file size, format, duration
- Timing Breakdown: load time, transcription time, words/second
- Processing Details: audio segments, audio processing time, words generated

## Standards

- Type hints required on all functions
- pytest for testing (fixtures in `tests/conftest.py`)
- PEP 8 with 100 character lines
- pylint for static code analysis

## Commands

```bash
# Setup
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```
