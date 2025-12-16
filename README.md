# Automatic Speech Recognition (ASR) Pipeline
Transcribe audio files to Markdown text.

## Installation
- Install [Python](https://www.python.org/downloads/)
- Install [FFmpeg](https://ffmpeg.org/). Use [Homebrew](https://brew.sh/) on macOS.

Run the following commands in the terminal.

- Set up a Python virtual environment: `python3 -m venv asr_pipeline_env`
- Activate the virtual environment: `source asr_pipeline_env/bin/activate` (Mac)
- Install the required Python packages: `pip install -r requirements.txt`
- Run the application in a web browser: `streamlit run streamlit_app.py`

## Notes
- Only .mp3 audio files are accepted. File size is limited to 2 MB.
- Transcript is exported to Markdown text.
