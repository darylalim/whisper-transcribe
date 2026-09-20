# Dependency history

How the `streamlit` floor in `pyproject.toml` moved — `>=1.58` while the app in fact ran on 1.57.0, then `>=1.59` for `st.skeleton`, then a support boundary again once the remote tabs went — and how the `mlx` stub situation was bisected release by release, with the commits behind the `ty` suppression that came and went. Moved out of `CLAUDE.md` so it is read when a floor is about to change or `mlx` is about to be re-locked, rather than loaded into every session. The live rules stay in `CLAUDE.md`: the `streamlit` bullet under **Dependencies** keeps the current floor, the `uv run --with 'streamlit==1.58.0' pytest` result (174/174) that makes it a boundary rather than a minimum, the `results_col.get("download_button")` form that keeps the *suite* at the same floor as the app, the rule that lowering a floor is its own decision rather than a side effect of deleting a feature, and the rule that changing the floor means `uv lock` in the same commit; the `import mlx.core as mx` paragraph under **Model** keeps the per-release stub inventory (present in 0.31.2 and 0.32.1+, absent in 0.32.0), the both-directions coupling of the `ty: ignore` directive to the locked version, and the `#`-prefix trap in the source comment. The pointers here sit in those two places, plus a third on the `st.skeleton` bullet under **Lazy Container Execution**, which sends a reader here for the bisect. Read this before lowering or raising the Streamlit floor, before re-locking `mlx` onto a release whose stubs have not been checked, and before restating why the suppression was added or removed.

## Dependencies: how the Streamlit floor moved

From the `streamlit` bullet under **Dependencies** — the one opening **`streamlit` — web UI (floor `>=1.59`; developed and tested against the locked 1.63.0)**. The bullet used to carry the whole history below inline; it now states the current position and points here. The four sub-sections are the sequence in order: the floor as a boundary over an older app, the one API that made it a minimum, the removal that made it a boundary again, and the test draft that nearly moved the minimum into the suite.

### `>=1.58`: a support boundary while the app ran on 1.57.0

**The floor was a real API minimum for a while, and is a support boundary again.** It sat at `>=1.58` as a deliberate support boundary while the app in fact ran on 1.57.0 — `st.tabs(key=, on_change=)` and `TabContainer.open` (the class in `mutable_tab_container.py`) were already there, and so are `st.space`, `st.badge`, `st.segmented_control(required=)` and `text_alignment`.

### `>=1.59`: the `st.skeleton` bisect

This is the bisect the `st.skeleton` bullet under **Lazy Container Execution** points to. `st.skeleton` was the one thing that is not: it lands in **1.59.0** (absent in 1.58.0, bisected directly), and adding it as the remote tabs' loading placeholder is what moved the floor.

### A boundary again: 174/174 at 1.58.0

**Those tabs are gone, and with them the app's only use of a 1.59 API** — the full suite passes with `uv run --with 'streamlit==1.58.0' pytest` (174/174, re-measured after the sidebar + two-column layout landed), so `>=1.59` is a support boundary again rather than an API minimum. The floor was left where it was all the same; the `streamlit` bullet in `CLAUDE.md` keeps the reason (lowering a floor is its own decision, not a side effect of deleting a feature) and the instruction to drop it to `>=1.58` deliberately if that is wanted.

### The first draft that nearly pinned the suite to 1.59

**That property nearly went with the layout change and was kept on purpose**: `Block.download_button` in `streamlit/testing/v1/element_tree.py` lands in 1.59.0, and a first draft of `test_main_area_splits_input_and_results_into_two_columns` read `results_col.download_button` — which would have made the *suite* need 1.59 while the app did not. It reads `results_col.get("download_button")` instead, the form `test_results_render_download_button_with_icon` already used; `at.sidebar`, `at.main`, `Block.columns`, `Block.caption`, `Block.status`, `Column.weight` and the `flex_container` block proto, which the other placement cases use, exist at 1.58.0 and 1.59.0 alike (checked by running the suite against both).

### Why `blinker` and `tenacity` left `uv.lock`

The same `streamlit` bullet ends with a parenthetical pointing here for this. `blinker` and `tenacity` left `uv.lock` with the 1.61.1 → 1.63.0 bump, not by a manual edit: 1.62.0 dropped `tenacity` (replaced by `streamlit/connections/retry_util.py`) and 1.63.0 dropped `blinker` (replaced by `streamlit/signal_util.py`); nothing here imported either.

## Model: how the `mlx` stubs were bisected

From the paragraph under **Model** opening **Note `import mlx.core as mx` carries no `ty` suppression, and it used to** — the one that records that `mlx/core` is a compiled `.so` which `ty` introspects only through the `mlx/core/*.pyi` stub package, and that whether a release carries those stubs is a per-release packaging property rather than a version floor. That paragraph keeps the inventory and the coupling rule; this section keeps the commits and the wrong generalisation an earlier draft drew from a single stub-less release.

### The bisect, and the commits behind the suppression

"This paragraph" in the text below is that **Model** paragraph in `CLAUDE.md`, an earlier version of which drew the generalisation named here. Checked by installing each: present in 0.31.2 and 0.32.1+, absent in 0.32.0 — the release `5ce3bdb`'s dependency bump landed on, which is why `ea15fec` added `# ty: ignore[unresolved-import]` four days later and an earlier version of this paragraph generalised that one gap into "mlx ships no `.pyi`" (the lock had been on 0.31.2, which `ty` resolves unaided). The 0.32.1 release notes never say "stubs" — the change hides under *Refactor wheel building script (#3818)*, closing ml-explore/mlx#3916 — so bisect by installing rather than by reading notes.

### The sequence, as commits

- The lock sat on **0.31.2**, which ships the stubs, so the bare `import mlx.core as mx` resolved and no suppression existed
- `5ce3bdb` (`chore(deps): upgrade all dependencies to latest`, 2026-08-08) moved the lock to **0.32.0**, the one release checked that ships no `mlx/core/*.pyi`, so the import became an `unresolved-import` error under `ty`
- `ea15fec` (`perf: reclaim MLX's allocator cache after each batch`, 2026-08-12, four days later) added `# ty: ignore[unresolved-import]` on the import line, and the `CLAUDE.md` paragraph of the time generalised the one gap into "mlx ships no `.pyi`" — false for 0.31.2 before it and for 0.32.1+ after it
- `5ee3b99` (`chore(deps): upgrade all dependencies to latest`, 2026-09-14) moved the lock to **0.32.2**, a stubbed release, so the *kept* directive became an `unused-ignore-comment` warning, `ty check` exits 1 on warnings, and the gate went red; the same commit removed the suppression (its `streamlit_app.py` hunk is that one line plus the comment above the import) — which is the "used to" in the **Model** paragraph's first sentence. The source comment above the import now names the directive without its leading `#` so the token stays inert; that is the parse trap the **Model** paragraph records, separate from this history
