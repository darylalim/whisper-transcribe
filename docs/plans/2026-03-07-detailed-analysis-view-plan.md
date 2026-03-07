# Detailed Transcript Analysis View — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add segment-level and word-level Whisper metrics to the UI and JSON export for research use.

**Architecture:** Replace Docling's `DocumentConverter` with direct `mlx_whisper.transcribe()` calls to access raw Whisper output (segment metrics, word probabilities). Split `_handle_transcription` into transcription (session state) and display (reads session state) to support interactive dataframe selection without losing results on Streamlit reruns.

**Tech Stack:** `mlx_whisper` (direct), Streamlit 1.54.0 (`st.dataframe` with `on_select`, `st.session_state`), `pandas`

**Design doc:** `docs/plans/2026-03-07-detailed-analysis-view-design.md`

---

## Key investigation result

Docling's pipeline strips all detailed metrics from `mlx_whisper` results. The raw `mlx_whisper.transcribe()` returns per-segment: `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, and per-word: `probability`. Docling discards all of these, keeping only timestamps and text. Therefore we must call `mlx_whisper.transcribe()` directly.

`mlx_whisper.ModelHolder` caches the loaded model in memory, so no need for `@st.cache_resource`.

Raw `mlx_whisper.transcribe()` result shape (from `mlx_whisper/transcribe.py:539`):

```python
{
    "text": "full transcript",
    "segments": [
        {
            "id": 0, "seek": 0,
            "start": 0.0, "end": 3.2,
            "text": " Hello world",
            "tokens": [50364, 2425, 1002, 50414],
            "temperature": 0.0,
            "avg_logprob": -0.25,
            "compression_ratio": 1.6,
            "no_speech_prob": 0.01,
            "words": [  # only when word_timestamps=True
                {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.98},
            ],
        }
    ],
    "language": "en",
}
```

---

## Shared test fixture

Used across tasks 1, 3, 4. Define at the top of `tests/test_app.py`:

```python
MOCK_WHISPER_RESULT = {
    "text": "Hello world",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.5,
            "text": " Hello world",
            "tokens": [50364, 2425, 1002, 50414],
            "temperature": 0.0,
            "avg_logprob": -0.25,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "words": [
                {"word": " Hello", "start": 0.0, "end": 1.0, "probability": 0.98},
                {"word": " world", "start": 1.0, "end": 2.5, "probability": 0.95},
            ],
        }
    ],
    "language": "en",
}
```

---

### Task 1: Replace Docling with direct mlx_whisper

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing tests for new `_transcribe`**

Replace the `mock_converter` fixture and old `_transcribe` tests with:

```python
@patch("streamlit_app.mlx_whisper")
def test_transcribe_success(mock_mlx):
    mock_mlx.transcribe.return_value = MOCK_WHISPER_RESULT

    result, elapsed = _transcribe(SAMPLE_AUDIO)

    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["avg_logprob"] == -0.25
    assert isinstance(elapsed, float)
    assert elapsed >= 0


@patch("streamlit_app.mlx_whisper")
def test_transcribe_no_text_raises(mock_mlx):
    mock_mlx.transcribe.return_value = {"text": "   ", "segments": [], "language": "en"}

    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(SAMPLE_AUDIO)
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_app.py::test_transcribe_success tests/test_app.py::test_transcribe_no_text_raises -v`
Expected: FAIL (mlx_whisper not imported, _transcribe has old signature)

**Step 3: Implement new `_transcribe` and remove Docling code**

In `streamlit_app.py`, replace:

```python
# REMOVE these imports:
from docling.datamodel import asr_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline
from streamlit.runtime.uploaded_file_manager import UploadedFile

# REMOVE these constants:
ASR_MODEL = asr_model_specs.WHISPER_TURBO_MLX
ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")

# REMOVE _get_converter function entirely
```

Add:

```python
import mlx_whisper
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-turbo"
```

Replace `_transcribe`:

```python
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
```

**Step 4: Update `_handle_transcription` minimally to consume new return type**

Change the line inside `_handle_transcription` from:
```python
transcript, eval_duration = _transcribe(tmp_path)
```
to:
```python
result, eval_duration = _transcribe(tmp_path)
transcript = result["text"].strip()
```

**Step 5: Update `_handle_transcription` tests for new mock return type**

Update the `@patch` decorators in all `test_handle_transcription_*` tests from:
```python
@patch("streamlit_app._transcribe", return_value=("Hello world", 1.23))
```
to:
```python
@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
```

Remove old tests and constants tests:
- Remove `test_asr_model_is_turbo`
- Remove `test_artifacts_path_is_absolute`
- Remove the `mock_converter` fixture

Update test imports from:
```python
from streamlit_app import (
    ARTIFACTS_PATH,
    ASR_MODEL,
    AUDIO_FORMATS,
    _get_audio_duration,
    _handle_transcription,
    _transcribe,
)
```
to:
```python
from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    _get_audio_duration,
    _handle_transcription,
    _transcribe,
)
```

Add a simple constant test:
```python
def test_asr_model_repo():
    assert ASR_MODEL_REPO == "mlx-community/whisper-turbo"
```

**Step 6: Run all tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: ALL PASS

**Step 7: Lint and commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add streamlit_app.py tests/test_app.py
git commit -m "Replace Docling with direct mlx_whisper for raw segment access"
```

---

### Task 2: Add `_format_timestamp` helper

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing tests**

```python
from streamlit_app import _format_timestamp

def test_format_timestamp_seconds():
    assert _format_timestamp(3.2) == "00:03.2"

def test_format_timestamp_minutes():
    assert _format_timestamp(65.3) == "01:05.3"

def test_format_timestamp_zero():
    assert _format_timestamp(0.0) == "00:00.0"
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_app.py::test_format_timestamp_seconds -v`
Expected: FAIL (ImportError)

**Step 3: Implement `_format_timestamp`**

Add to `streamlit_app.py` (before `_transcribe`):

```python
def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -k format_timestamp -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "Add _format_timestamp helper for MM:SS.s display"
```

---

### Task 3: Split display logic and add inner tabs

This task splits `_handle_transcription` into two functions:
- `_handle_transcription` — transcribes and stores result in `st.session_state`
- `_display_transcription` — reads from session state and renders UI

This split is needed because interactive widgets (dataframe selection) trigger Streamlit reruns. Without session state, the transcription output disappears on rerun since the "Transcribe" button is no longer pressed.

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing tests for `_handle_transcription` (session state)**

```python
@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_stores_result(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    assert "transcription" in mock_st.session_state
    data = mock_st.session_state["transcription"]
    assert data["result"] == MOCK_WHISPER_RESULT
    assert data["eval_duration"] == 1.23
    assert data["audio_duration"] == 10.5
    assert data["file_stem"] == "interview_transcript"


@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=None)
def test_handle_transcription_stores_null_duration(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.warning.assert_called_once()
    assert mock_st.session_state["transcription"]["audio_duration"] is None


@patch(
    "streamlit_app._transcribe",
    side_effect=RuntimeError("Transcription produced no text"),
)
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_runtime_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once()
    assert "transcription" not in mock_st.session_state


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_unexpected_error(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(mock_uploaded_file)

    mock_st.error.assert_called_once_with("Unexpected error: unexpected")
    mock_st.exception.assert_called_once()
```

**Step 2: Write failing tests for `_display_transcription`**

Update the `mock_st` fixture to support tabs and session_state:

```python
@pytest.fixture
def mock_st():
    with patch("streamlit_app.st") as m:
        m.spinner.return_value.__enter__ = MagicMock()
        m.spinner.return_value.__exit__ = MagicMock(return_value=False)
        m.session_state = {}
        tab1, tab2 = MagicMock(), MagicMock()
        tab1.__enter__ = MagicMock(return_value=tab1)
        tab1.__exit__ = MagicMock(return_value=False)
        tab2.__enter__ = MagicMock(return_value=tab2)
        tab2.__exit__ = MagicMock(return_value=False)
        m.tabs.return_value = (tab1, tab2)
        m.columns.return_value = (MagicMock(), MagicMock())
        mock_event = MagicMock()
        mock_event.selection.rows = []
        m.dataframe.return_value = mock_event
        yield m
```

```python
from streamlit_app import _display_transcription

def test_display_transcription_no_session_state(mock_st):
    _display_transcription()
    mock_st.tabs.assert_not_called()


def test_display_transcription_shows_caption_and_tabs(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "eval_duration": 1.23,
        "audio_duration": 10.5,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.caption.assert_called_once_with(
        "10.5s audio \u00b7 2 words \u00b7 transcribed in 1.23s"
    )
    mock_st.tabs.assert_called_once_with(["Transcript", "Detailed Analysis"])
    mock_st.code.assert_called_once_with(
        "Hello world", language=None, wrap_lines=True
    )


def test_display_transcription_without_duration(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "eval_duration": 1.23,
        "audio_duration": None,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    mock_st.caption.assert_called_once_with(
        "2 words \u00b7 transcribed in 1.23s"
    )
```

**Step 3: Run tests to verify failure**

Run: `uv run pytest tests/test_app.py -k "display_transcription or handle_transcription_stores" -v`
Expected: FAIL (_display_transcription doesn't exist yet)

**Step 4: Implement new `_handle_transcription` and `_display_transcription`**

Replace `_handle_transcription` in `streamlit_app.py`:

```python
def _handle_transcription(uploaded_file: UploadedFile) -> None:
    name = Path(uploaded_file.name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"audio{name.suffix}"
        tmp_path.write_bytes(uploaded_file.read())

        try:
            audio_duration = _get_audio_duration(tmp_path)
            if audio_duration is None:
                st.warning(
                    "Could not determine audio duration. Transcribing anyway."
                )

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
        " \u00b7 ".join(
            part
            for part in [
                f"{audio_duration:.1f}s audio"
                if audio_duration is not None
                else None,
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
    c1.download_button(
        "Download transcript", transcript, file_stem + ".txt", "text/plain"
    )
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
```

Add placeholder for `_show_detailed_analysis` (implemented in Task 4):

```python
def _show_detailed_analysis(segments: list[dict]) -> None:
    st.info("Detailed analysis coming soon.")
```

Update the UI section at the bottom of the file:

```python
# UI
st.title("Audio Transcription")
st.write("Upload or record audio to transcribe with Whisper.")

upload_tab, record_tab = st.tabs(["Upload", "Record"])
with upload_tab:
    uploaded_file = st.file_uploader("Upload audio file", type=AUDIO_FORMATS)
    if uploaded_file:
        st.audio(uploaded_file)
    upload_submitted = st.button(
        "Transcribe", type="primary", key="upload_btn", disabled=not uploaded_file
    )

with record_tab:
    recorded_audio = st.audio_input("Record audio")
    if recorded_audio:
        st.audio(recorded_audio)
    record_submitted = st.button(
        "Transcribe", type="primary", key="record_btn", disabled=not recorded_audio
    )

if record_submitted and recorded_audio:
    _handle_transcription(recorded_audio)
elif upload_submitted and uploaded_file:
    _handle_transcription(uploaded_file)

_display_transcription()
```

**Step 5: Remove old `_handle_transcription` tests**

Remove these tests (replaced by the new ones above):
- `test_handle_transcription_with_duration`
- `test_handle_transcription_without_duration`
- `test_handle_transcription_runtime_error`
- `test_handle_transcription_unexpected_error`
- `test_handle_transcription_temp_dir_cleanup`

Add replacement temp dir cleanup test:

```python
@patch("streamlit_app._transcribe", return_value=(MOCK_WHISPER_RESULT, 1.23))
@patch("streamlit_app._get_audio_duration", return_value=10.5)
def test_handle_transcription_temp_dir_cleanup(
    mock_duration, mock_transcribe, mock_st, mock_uploaded_file
):
    called_paths = []
    mock_duration.side_effect = lambda p: (called_paths.append(p), 10.5)[1]

    _handle_transcription(mock_uploaded_file)

    assert len(called_paths) == 1
    assert called_paths[0].name == "audio.mp3"
    assert not called_paths[0].exists()
```

**Step 6: Run all tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: ALL PASS

**Step 7: Lint and commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add streamlit_app.py tests/test_app.py
git commit -m "Split display logic with session state, add inner tabs"
```

---

### Task 4: Add segment dataframe and word detail

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing tests for `_show_detailed_analysis`**

```python
from streamlit_app import _show_detailed_analysis

def test_show_detailed_analysis_renders_segment_dataframe(mock_st):
    _show_detailed_analysis(MOCK_WHISPER_RESULT["segments"])

    assert mock_st.dataframe.called
    call_args = mock_st.dataframe.call_args_list[0]
    df = call_args[0][0]
    assert "#" in df.columns
    assert "Start" in df.columns
    assert "End" in df.columns
    assert "Text" in df.columns
    assert "Avg Log Prob" in df.columns
    assert "No Speech Prob" in df.columns
    assert "Compression Ratio" in df.columns
    assert "Temperature" in df.columns
    assert len(df) == 1
    assert df.iloc[0]["#"] == 0
    assert df.iloc[0]["Avg Log Prob"] == -0.25


def test_show_detailed_analysis_no_segments(mock_st):
    _show_detailed_analysis([])

    mock_st.info.assert_called_once_with("No segment detail available.")
    mock_st.dataframe.assert_not_called()


def test_show_detailed_analysis_with_row_selected(mock_st):
    mock_event = MagicMock()
    mock_event.selection.rows = [0]
    mock_st.dataframe.return_value = mock_event

    _show_detailed_analysis(MOCK_WHISPER_RESULT["segments"])

    # Two dataframe calls: segment table + word table
    assert mock_st.dataframe.call_count == 2
    word_df = mock_st.dataframe.call_args_list[1][0][0]
    assert "Word" in word_df.columns
    assert "Probability" in word_df.columns
    assert len(word_df) == 2
    assert word_df.iloc[0]["Word"] == "Hello"
    assert word_df.iloc[0]["Probability"] == 0.98
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_app.py -k "show_detailed" -v`
Expected: FAIL (placeholder implementation)

**Step 3: Implement `_show_detailed_analysis`**

Replace the placeholder in `streamlit_app.py`:

```python
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
        use_container_width=True,
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
            st.dataframe(
                pd.DataFrame(word_data), use_container_width=True
            )
```

**Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: ALL PASS

**Step 5: Lint and commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add streamlit_app.py tests/test_app.py
git commit -m "Add segment dataframe and word detail in Detailed Analysis tab"
```

---

### Task 5: Add JSON export test and update docs

**Files:**
- Modify: `tests/test_app.py`
- Modify: `CLAUDE.md`

**Step 1: Write test for JSON with segments**

```python
def test_display_transcription_json_includes_segments(mock_st):
    mock_st.session_state["transcription"] = {
        "result": MOCK_WHISPER_RESULT,
        "eval_duration": 1.23,
        "audio_duration": 10.5,
        "file_stem": "interview_transcript",
    }

    _display_transcription()

    col1, col2 = mock_st.columns.return_value
    payload = json.loads(col2.download_button.call_args[0][1])
    assert payload["audio_duration"] == 10.5
    assert payload["transcript"] == "Hello world"
    assert payload["num_words"] == 2
    assert payload["eval_duration"] == 1.23
    assert "segments" in payload
    assert len(payload["segments"]) == 1
    seg = payload["segments"][0]
    assert seg["index"] == 0
    assert seg["avg_logprob"] == -0.25
    assert seg["no_speech_prob"] == 0.01
    assert seg["compression_ratio"] == 1.2
    assert seg["temperature"] == 0.0
    assert len(seg["words"]) == 2
    assert seg["words"][0]["word"] == "Hello"
    assert seg["words"][0]["probability"] == 0.98
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: ALL PASS (JSON export was implemented in Task 3)

**Step 3: Update `CLAUDE.md`**

Update the following sections to reflect the new architecture:
- **Model**: Remove Docling reference, note direct `mlx_whisper` usage
- **Performance**: Replace `@st.cache_resource` note with `mlx_whisper.ModelHolder` caching
- **Architecture**: Add `_display_transcription`, `_show_detailed_analysis`, `_format_timestamp`; note session state usage; remove `_get_converter`, `ARTIFACTS_PATH`
- **JSON Download**: Add `segments` array with nested fields
- **Testing**: Update test descriptions

**Step 4: Run lint**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: clean

**Step 5: Commit**

```bash
git add tests/test_app.py CLAUDE.md
git commit -m "Add JSON segment export test and update docs"
```
