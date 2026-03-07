# Detailed Transcript Analysis View

## Goal

Add word-level and segment-level Whisper output to the UI and JSON export, serving AI researchers who need granular transcription data for studying model robustness, biases, and constraints.

## Data Layer

Modify `_transcribe` to return structured data instead of just markdown text. The return value becomes a dict:

```json
{
  "transcript": "full text",
  "segments": [
    {
      "index": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello world...",
      "avg_logprob": -0.25,
      "no_speech_prob": 0.01,
      "compression_ratio": 1.6,
      "temperature": 0.0,
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4, "probability": 0.98}
      ]
    }
  ]
}
```

**Pre-implementation investigation required:** Confirm how Docling's conversion result object exposes the raw Whisper segments and word-level data.

## UI Layout

After transcription, the output area becomes:

```
[Upload | Record]                    (existing input tabs)
  audio preview
  [Transcribe]

  "12.3s audio · 150 words · transcribed in 2.45s"   (metrics caption, unchanged)

  [Transcript | Detailed Analysis]   (new inner tabs)
    (tab content)

  [Download transcript.txt]  [Download JSON]   (download buttons, unchanged position)
```

### Transcript Tab

Same as current: `st.code` with full text, wrap enabled.

### Detailed Analysis Tab

**Segment table** — `st.dataframe` with columns:
- `#` (segment index)
- `Start` / `End` (formatted as `MM:SS.s`)
- `Text` (truncated for scannability)
- `Avg Log Prob`
- `No Speech Prob`
- `Compression Ratio`
- `Temperature`

**Word detail** — Row selection via `st.dataframe` `on_select` (Streamlit 1.54.0 supports this) reveals a word-level table below:
- `Word`
- `Start` / `End`
- `Probability`

## JSON Export

Extends the current format with a `segments` array. Existing top-level fields preserved for backward compatibility:

```json
{
  "audio_duration": 12.3,
  "transcript": "Full transcript text...",
  "num_words": 150,
  "eval_duration": 2.45,
  "segments": [...]
}
```

## Error Handling

- If segment/word data is unavailable from Docling, fall back gracefully: plain transcript in both tabs, `st.info` in Detailed Analysis tab, JSON without `segments` key.

## Testing

- Update `_transcribe` tests for new structured return format
- Add tests for segment/word data extraction from Docling result
- Add tests for Detailed Analysis tab rendering (mocked `st.dataframe`)
- Add tests for JSON download containing full hierarchy
