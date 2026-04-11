import json
import subprocess
import tempfile
import time
from pathlib import Path

import mlx_whisper
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
AUDIO_FORMATS = ("aac", "flac", "m4a", "mov", "mp3", "mp4", "ogg", "wav", "webm")


def _get_audio_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


def _transcribe(path: Path) -> tuple[dict, float]:
    start = time.perf_counter()
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=ASR_MODEL_REPO,
        language="en",
        task="transcribe",
        word_timestamps=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    elapsed = round(time.perf_counter() - start, 2)
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result, elapsed


def _handle_transcription(uploaded_file: UploadedFile) -> None:
    name = Path(uploaded_file.name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"audio{name.suffix}"
        tmp_path.write_bytes(uploaded_file.read())

        try:
            audio_duration = _get_audio_duration(tmp_path)
            if audio_duration is None:
                st.warning("Could not determine audio duration. Transcribing anyway.")

            with st.spinner("Transcribing..."):
                result, eval_duration = _transcribe(tmp_path)

            st.session_state["transcription"] = {
                "result": result,
                "eval_duration": eval_duration,
                "audio_duration": audio_duration,
                "file_stem": name.stem + "_transcript",
            }
        except RuntimeError as e:
            st.error(f"Transcription failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.exception(e)


def _show_detailed_analysis(segments: list[dict]) -> None:
    if not segments:
        st.info("No segment detail available.")
        return

    segment_data = [
        {
            "#": i,
            "Start": _format_timestamp(seg["start"]),
            "End": _format_timestamp(seg["end"]),
            "Text": seg["text"].strip(),
            "Avg Log Prob": round(seg["avg_logprob"], 4),
            "No Speech Prob": round(seg["no_speech_prob"], 4),
            "Compression Ratio": round(seg["compression_ratio"], 2),
            "Temperature": seg["temperature"],
        }
        for i, seg in enumerate(segments)
    ]
    df = pd.DataFrame(segment_data)
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
    )

    if event.selection.rows:
        sel_idx = event.selection.rows[0]
        words = segments[sel_idx].get("words", [])
        if words:
            word_data = [
                {
                    "Word": w["word"].strip(),
                    "Start": _format_timestamp(w["start"]),
                    "End": _format_timestamp(w["end"]),
                    "Probability": round(w["probability"], 4),
                }
                for w in words
            ]
            st.dataframe(pd.DataFrame(word_data), width="stretch")


def _display_transcription() -> None:
    if "transcription" not in st.session_state:
        return

    data = st.session_state["transcription"]
    result = data["result"]
    eval_duration = data["eval_duration"]
    audio_duration = data["audio_duration"]
    file_stem = data["file_stem"]

    transcript = result["text"].strip()
    segments = result.get("segments", [])
    num_words = len(transcript.split())

    st.caption(
        " · ".join(
            part
            for part in [
                f"{audio_duration:.1f}s audio" if audio_duration is not None else None,
                f"{num_words:,} words",
                f"transcribed in {eval_duration:.2f}s",
            ]
            if part
        )
    )

    transcript_tab, detail_tab = st.tabs(["Transcript", "Detailed Analysis"])
    with transcript_tab:
        st.code(transcript, language=None, wrap_lines=True)
    with detail_tab:
        if segments:
            _show_detailed_analysis(segments)
        else:
            st.info("No segment detail available.")

    c1, c2 = st.columns(2)
    c1.download_button("Download transcript", transcript, file_stem + ".txt", "text/plain")
    c2.download_button(
        "Download JSON",
        json.dumps(
            {
                "audio_duration": audio_duration,
                "transcript": transcript,
                "num_words": num_words,
                "eval_duration": eval_duration,
                "segments": [
                    {
                        "index": i,
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                        "temperature": seg["temperature"],
                        "avg_logprob": seg["avg_logprob"],
                        "compression_ratio": seg["compression_ratio"],
                        "no_speech_prob": seg["no_speech_prob"],
                        "words": [
                            {
                                "word": w["word"].strip(),
                                "start": w["start"],
                                "end": w["end"],
                                "probability": w["probability"],
                            }
                            for w in seg.get("words", [])
                        ],
                    }
                    for i, seg in enumerate(segments)
                ],
            },
            indent=2,
        ),
        file_stem + ".json",
        "application/json",
    )


# UI
st.title("Whisper Pipeline")
record_tab, upload_tab = st.tabs(["Record", "Upload"])
with record_tab:
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    if recorded_audio:
        st.audio(recorded_audio)
    record_submitted = st.button(
        "Transcribe", type="primary", key="record_btn", disabled=not recorded_audio
    )

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload audio file", type=AUDIO_FORMATS, label_visibility="collapsed"
    )
    if uploaded_file:
        st.audio(uploaded_file)
    upload_submitted = st.button(
        "Transcribe", type="primary", key="upload_btn", disabled=not uploaded_file
    )

if record_submitted and recorded_audio:
    _handle_transcription(recorded_audio)
elif upload_submitted and uploaded_file:
    _handle_transcription(uploaded_file)

_display_transcription()
