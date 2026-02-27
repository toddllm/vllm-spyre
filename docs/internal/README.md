# Internal Docs Usage

This folder is for working notes that help active development and testing.

## Why this exists
- Keep upstream-facing docs under `docs/roadmaps` clean and reviewable.
- Preserve detailed lab notes, host-specific troubleshooting, and iteration
  history without losing context.
- Make it easier to cut focused PRs from a long-running integration branch.

## Conventions
- Put iterative notes in `docs/internal/lab_notes/`.
- Keep language environment-agnostic where possible (`local CPU`, `remote CUDA`)
  instead of machine-specific hostnames.
- If a note is only relevant to a temporary local setup, prefer this folder
  over public roadmap docs.

## PR hygiene for this branch
- Treat production code and tests in `vllm_spyre/` and `tests/v1/worker/` as the
  default PR payload.
- Treat `docs/internal/` as optional context; do not include it in upstream PRs
  unless explicitly requested.
- For targeted PRs, cherry-pick code/test commits and skip note-only commits.
