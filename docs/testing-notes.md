# Testing notes

The per-mutation failure maps, the order-dependence repro and the earlier states of the test suite, kept so none of them is re-derived or re-measured by accident. Moved out of `CLAUDE.md` so they are read on demand — when a mutation check is re-run, when a green suite is being read as evidence of cache isolation, or when the failure count of an `_error` mutation is about to be restated — rather than loaded into every session. Every entry here was cut from the **Testing** section of `CLAUDE.md`, which keeps the live rules (what each case pins, which fixture isolates what, and the standing instruction to re-measure rather than reason) and points back to this file for the `_clear_caches` repro, the placement failure map and the `_escape_markdown` failure-count sequence (the sidebar and results-column bullets under **Input Modes** and **Error Handling** point here too, for the mutation lists). The other entries carry no pointer: two were condensed in place without naming this file — the `columns` stub and the `file_uploader` claim; two more — the whitespace-collapse map and `_plural` — keep their maps in full in `CLAUDE.md`, and this file holds only the earlier-version clauses cut from them; and three were cut outright rather than summarised — the tab-gate history under the `_clear_caches` fixture, the earlier states of the source-dispatch mutation check, and the YouTube `RuntimeError` case under `_error` — so for those five, and for the two cut clauses, this file is the only record; the entries below are in `CLAUDE.md` order, each headed so the pointer's promise — the repro, the sequence, the failure map — is findable, and each names the bullet it came from. Nothing was deleted in the move: every sentence, number and case name is reproduced as it stood, and where the text says "this file", "this line" or "here" it means `CLAUDE.md` and the bullet it was cut from, as each lead-in says.

## The autouse `_clear_caches` fixture: the order-dependence repro

From the opening paragraph of **Testing** (`Mocked at the boundary (`mlx_whisper`, `st`)…`), on why the fixture's `st.cache_data.clear()` is load-bearing beside the per-wrapper `_transcribe.clear()`, and why only file order hides it. The repro, which the `CLAUDE.md` pointer summarises as "run after the fallback case with the clear removed, it reads the fallback case's cached result … and fails":

Replacing the global clear with `pass` leaves the full suite green (measured: 174/174 after the sidebar + two-column layout landed; 163/163 before it) — but `test_transcription_failure_renders_an_escaped_alert` seeds the same `b"upload bytes"` / `.mp3` / default kwargs as `test_transcribe_falls_back_to_a_loaded_upload_from_an_empty_tab` and `test_transcript_format_drives_include_subtitles`, and `_transcribe`'s cache key does not include the filename. Run the fallback case first and then the failure case with the clear removed, and the failure case reads the fallback case's cached result from the AppTest-module `_transcribe`, never raises, and fails with `assert [] == ['Transcription failed for …']` (measured by passing the two node ids in that order). It passes in a normal run only because it is defined earlier in the file. That is the order-*independence* the global clear buys, and it now has a live subject rather than a hypothetical one.

The global clear had an earlier subject as well, which went with the YouTube and URL tabs. It used to guard the remote tabs' tab-gate cases the same way, whose `call_count == 0` assertions would have passed on a stale hit even with the `tab.open` guard removed.

The fixture's `st.cache_resource.clear()` nearly went with those tabs: a first draft of the change that removed them dropped the call, in favour of a "restore it when the first `cache_resource` function comes back" rule, on the grounds that nothing was `@st.cache_resource` any more. It was put back because that rule would have re-created the silent order-dependence above with nothing to fail until someone remembered it — the standing rule under **Testing** (a fixture that isolates caches isolates both stores regardless of what the app uses) is what replaced the draft.

## `mock_st`'s `columns` stub: why it is unexercised, then and now

From the paragraph beginning **`mock_st` is not a bare `MagicMock`**. The `CLAUDE.md` paragraph keeps the current reason and carries no pointer here; this is the full account, including the one it replaced:

**The `columns` stub is still unexercised, and the reason has changed.** It used to be that nothing in `streamlit_app.py` called `st.columns` at all; now the module body calls `st.columns(2, gap="small")` exactly once, for the input | results split, but that line runs only under `AppTest`, which executes the real `st`. None of the functions the mocked-`st` cases call — `_handle_transcription`, `_display_transcription`, `_error` — calls `st.columns`, so the stub is never hit. It is kept precisely because the failure it prevents is unreadable: the first `st.columns` inside a function under test gets a working fixture instead of a `MagicMock` unpacking traceback — but do not read its presence as evidence that any *function* uses columns; the app's one call is module-level and outside the fixture's reach.

## The whitespace collapse: the two earlier-version clauses

From the end of the `_handle_transcription` bullet under **Testing** (`session-state storage, per-file error handling (RuntimeError + unexpected), kwarg forwarding…`), which still carries the whole map — the `st.error`-sink case, `test_display_transcription_collapses_whitespace_in_subheader` at the heading sink, and which three of the four cases reverting the helper's last line to `sub(r"\\\1", text)` fails. `CLAUDE.md` is the copy to update; only two clauses were cut, and they are the history behind that map:

- The heading case is a separate case not only because the heading is a separate renderer but *because an earlier version of `CLAUDE.md` wrongly claimed the heading was already protected* — the "hardened for free" claim, whose refutation is in `docs/design-history.md` under `_escape_markdown`.
- The three-of-four figure is measured; *an earlier version of that line said all four*. The `st.error` case is the one that survives, because `_condense` runs first in `_error` and collapses whitespace itself — still stated in `CLAUDE.md`, beside the figure.

## `_plural`: where the second mutation used to point

From the **Formatting helpers** bullet's `_plural` entry under **Testing**, which still carries both mutations in full — `count > 1` fails the zero case and nothing else; an unconditional `f"{count} {noun}s"` gets zero *right* and fails the singular case plus every single-file status-label assertion, four cases. `CLAUDE.md` is the copy to update. The one clause cut: *an earlier version of this line pointed the second mutation at the zero case* — which that mutation does not fail, so the line was describing a check that would have passed with the helper broken.

## Source dispatch: the earlier states of the `AppTest` mutation check

From the **Source dispatch** bullet (`two `AppTest` cases and a unit test, and the `AppTest` pair are deliberately **not** symmetric`). "This bullet" is that one, the mutation check it refers to is `test_transcribe_runs_the_open_tab_over_a_loaded_upload`, and the unit test is `test_active_sources`:

A first draft of this bullet claimed the `AppTest` mutation check was impossible and made the unit test carry it — the review refuted that with the `audio_input` patch. (Before the remote tabs were removed the `AppTest` mutation check was a URL tab open with an upload still loaded, and nothing covered dispatch at all before that, which is why a 146-test suite stayed green with the bug in place)

The unit test's `none_reads_as_closed` case had a first draft too: it shipped as an all-`None` layout, which pins nothing — with every flag `None` the "closed" and "open" readings both yield the priority order, so a None-as-open implementation passes it. The case now uses the deliberately *mixed* layout `[(None, upload), (True, recording)]` → the recording, where None-as-open returns the upload instead (measured: `is_open is not False` fails it); the test's comment carries the rationale, and the vacuous first draft is recorded only here.

## `st.audio`'s `format=` wiring: the stale "uncoverable" claim about `file_uploader`

From the **`st.audio`'s `format=` wiring, pinned through the media URL** bullet, on `test_upload_preview_renders_with_its_declared_mime`. "This" is the `format=` assertion and "this file" is `CLAUDE.md`; the `CLAUDE.md` bullet keeps only the `FileUploader.set_value` facts and carries no pointer here, and this is the history behind them:

This used to be pinned only on the two remote tabs' previews, and this file long asserted the Upload site was uncoverable because `AppTest` could not seed a `file_uploader`. That claim was stale: since at least the 1.59 floor — checked directly against 1.59.0 — `FileUploader.set_value` takes `(name, bytes, mime)`, or a sequence of those under `accept_multiple_files=True` (`streamlit/testing/v1/element_tree.py`, accessor `at.file_uploader`). `_upload` wraps it, and every `AppTest` case that needs **Transcribe** enabled goes through it. It is present at the floor with the same signature, and CI runs the locked 1.63.0, so the use is safe at both ends of the range

## `_error`: the real-`st.error` case, the failure count's sequence, and the case that does not join it

Three spans from the **`_error` is tested directly** bullet, in the order they sat there.

### `test_transcription_failure_renders_an_escaped_alert`'s first end-to-end pin

The bullet names that case as the one that pushes an `_error` message through the *real* `st.error`. It also gives `_handle_transcription`'s failure replay its first end-to-end pin; the YouTube `RuntimeError` case used to be the only real-`st.error` assertion, and went with its tab.

### The sequence of the `_escape_markdown` failure count

The count is how many cases fail when the `_escape_markdown` call inside `_error` is dropped — **12** as of the sidebar + two-column layout — and the `CLAUDE.md` pointer says it "has moved four times already (the sequence is in `docs/testing-notes.md`)". The sequence:

**This count is a maintenance trap and has now moved four times** — 8 until the YouTube `RuntimeError` `AppTest` case, 9 until `_condense` arrived, 11 with both, 10 when the YouTube case went with its tab, 12 with the scheme-only and real-`st.error` cases. Any new exact-match assertion on an `_error` message moves it again, so re-measure rather than reason: mutate the call and read the failure count. Re-measured after the sidebar + two-column layout landed: still **12**.

### The time-range case that does not join the count

`test_time_range_error_renders_beside_the_transcribe_button` does read an `_error` alert end-to-end, but it compares against the raw `_validate_time_range("90,30")` string, which carries no character in the escape class, so — like `test_invalid_time_range_shows_inline_error` before it — it passes with the escape dropped and does not join the count; the column-scoped rewrite of `test_transcription_failure_renders_an_escaped_alert` kept its escaped string and stays in it

## Placement: the per-mutation failure map

From the paragraph beginning **Placement is pinned through `at.sidebar` / `at.main`, `Block.columns` and the block protos**, which lists the placement cases; the `CLAUDE.md` pointer names the mutations, and this is the map of which cases each one fails. "Here" in its last sentence is the **Testing** section of `CLAUDE.md`:

Every one is mutation-checked, each applied to a copy with the suite run and the file restored byte-identical: `with st.sidebar:` → `with st.container():` fails six cases — the sidebar case, the two `at.main.subheader` reads, and the tiers, help and required cases, i.e. everything that reads `at.sidebar`; `with results_col:` → `with input_col:` fails four — the two-columns case, the empty-hint case, the column-scoped failure-alert case and the status-column case; hoisting `_handle_transcription` out of `with results_col:` fails the status case and the column-scoped failure-alert case; `st.columns(2)` → `st.columns([2, 3])` fails the weights assertion; `"right"` → `"distribute"` on the Transcribe container fails the alignment case; deleting one seam fails the tiers case; dropping one `help=` fails the help case; dropping `required=True` fails the required case; deleting the empty-state hint fails the empty-hint case and the failure-alert case (whose second `at.run()` expects the hint back); `layout` back to `"centered"` fails `test_page_config` only, because `AppTest` renders no geometry — the column widths, the fold, the hint threshold and the sidebar wrap are browser measurements and live under **Input Modes** and **Accepted formats**, not here.
