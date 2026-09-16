# Whisper Transcribe

[![CI](https://github.com/darylalim/whisper-transcribe/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/whisper-transcribe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Transcribe and translate audio and video **locally on your Mac** — no cloud, no uploads, no cost. This Streamlit application is powered by OpenAI's Whisper and accelerated on Apple Silicon with MLX (Apple's machine-learning framework). Bring your own files or record straight from the browser.

![Whisper Transcribe in its dark macOS-style theme — a Settings sidebar on the left holding the primary-language selector, a translate toggle, a Plain text / Subtitles transcript-format control, a no-verbatim toggle, and a collapsed Advanced options panel; the main area shows the two input tabs (Upload, Record) with the upload dropzone and a Transcribe button on the left, and an empty results panel reading "Transcripts appear here" on the right](docs/screenshot.png)

## Features

- **[OpenAI Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)** via [mlx-whisper](https://pypi.org/project/mlx-whisper/), accelerated on Apple Silicon
- **On-device processing** — audio is transcribed entirely on your machine; nothing is uploaded (only the one-time model-weights download on first run uses the network)
- **100-language transcription** with auto-detect or manual selection
- **Translate non-English audio to English**
- **Two input modes** — multi-file upload (up to 500 MB per file) and in-browser recording
- **Editable subtitle preview**, exportable as SRT (the standard subtitle file format)
- **No verbatim** — skips hallucinated text over music, applause, and other non-speech audio
- **Decode independently** — each 30 s window is transcribed without prior-window context; more robust on noisy or music-heavy audio
- **Time-range clipping** — transcribe only selected portions (comma-separated `start,end` pairs in seconds)
- **Keyterms** — bias decoding toward proper nouns and jargon (up to 50 terms)
- **Instant repeat results** — identical file-and-settings combinations are served from cache
- **macOS-style light and dark themes** — Apple's system greys and accent blue, so the app looks like the Mac it runs on, with Material Symbol icons; pick System, Light or Dark from the ⋮ menu in the top-right corner (Streamlit's own menu, not the app's Settings sidebar)

## How it works

You provide audio or video through one of two tabs (upload or record) and set any options in the sidebar. The app writes the audio to a temporary file and runs `mlx_whisper.transcribe()` with the Whisper large-v3-turbo model locally on Apple Silicon via MLX. The result is cached and rendered beside the input as editable plain text (or SRT when subtitles are enabled), and can be downloaded as `.txt` or `.srt`. See [CLAUDE.md](CLAUDE.md) for the full architecture.

![A completed transcription in Whisper Transcribe's dark theme — the Settings sidebar and, in the main area, an uploaded file's audio player with an enabled blue Transcribe button on the left; on the right, a "Transcribed 1/1 file" status above a bordered result section holding the filename, the editable transcript, and a Download button to save it as .txt or .srt](docs/screenshot-result.png)

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.12+
- [FFmpeg](https://formulae.brew.sh/formula/ffmpeg)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/darylalim/whisper-transcribe.git
cd whisper-transcribe
brew install ffmpeg
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

Upload one or more files (audio: `mp3, m4a, wav, opus`; video: `mp4, mov, webm, mkv`) or record audio in-browser, then click **Transcribe**. Transcripts appear beside the input, one editable section per file. The app opens in Streamlit's wide layout with the settings in a sidebar and is laid out for a desktop window: the dropzone's format list is shown in full when the window is at least about 1460 px wide with the sidebar open (in a narrower window it is abbreviated with an ellipsis — every format is still accepted — and collapsing the sidebar restores it down to about 1160 px); below ~770 px the sidebar collapses on its own, and at 640 px or narrower the two panels stack.

> **First run:** the first time you transcribe, the Whisper large-v3-turbo weights (~1.5 GB) are downloaded from Hugging Face and cached locally, so the first transcription takes longer and needs an internet connection. Subsequent transcriptions run offline.

Optional controls, in the **Settings** sidebar:

- **Primary language** — auto-detected by default
- **Translate to English** — translate non-English audio
- **Transcript format** — choose **Plain text** or **Subtitles**; picking Subtitles shows an editable SRT subtitle preview and switches the **Download** button from `.txt` to `.srt`. Subtitle cues are wrapped to 42 characters per line, the broadcast convention, with the lines balanced rather than one filled and one left short
- **No verbatim** — skip silent stretches where Whisper appears to be hallucinating text, such as over music or applause after speech ends; it does not remove filler words or repetitions
- **Decode independently** — disable prior-window context; more robust on noisy or music-heavy audio, at the cost of slightly choppier wording where 30 s windows meet
- **Time range** — transcribe only selected portions; comma-separated `start,end` pairs in seconds (e.g., `30,90` for one clip, `0,60,120,180` for multiple); an invalid range disables **Transcribe** and is flagged beside it
- **Keyterms** — bias decoding toward specific terms (proper nouns, jargon)

**Decode independently**, **Time range**, and **Keyterms** are grouped under an **Advanced options** expander at the bottom of the sidebar. An invalid time range is flagged directly above the **Transcribe** button it disables, so the reason stays visible while the expander is collapsed.

## Development

After `uv sync`, the project's checks run through uv:

```bash
uv run pytest             # run the test suite
uv run ruff check .       # lint
uv run ruff format .      # format
uv run ty check           # type-check
```

CI runs the same tools on every push to `main` and on pull requests — it uses `ruff format --check .` to *verify* formatting rather than apply it, so run `ruff format .` locally before committing. The workflow targets a **macos-14 (Apple Silicon) runner** because of the compute backend, not wheel availability: `mlx` pulls in a backend automatically only on macOS (`mlx-metal`), whereas on Linux it is an opt-in extra — so a Linux runner installs cleanly and then fails at `import mlx_whisper`. Please make sure `ruff`, `ty`, and `pytest` pass before opening a pull request.

## Troubleshooting

- **`ffmpeg` not found** — install it with `brew install ffmpeg` and verify with `ffmpeg -version`. FFmpeg is required for decoding audio and video.
- **Intel Mac / non–Apple Silicon** — `mlx-whisper` requires Apple Silicon and will not run on Intel Macs.
- **Long pause on the first transcription** — the model weights (~1.5 GB) are downloading from Hugging Face (see the *First run* note above); this needs a network connection and only happens once.
- **Download is missing your last edit** — the transcript box commits when it loses focus, so click outside it (or press Ctrl/Cmd+Enter) before pressing **Download**.

## License

This project is licensed under the [MIT License](LICENSE).
