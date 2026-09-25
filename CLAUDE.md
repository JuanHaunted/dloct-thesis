# DLOCT thesis: instructions for Claude sessions

Two Claude sessions work in this repo in parallel:

| Role | Owns | Does not touch |
|---|---|---|
| **Programming agent** | `src/`, `configs/`, `scripts/`, `tests/`, `docs/status.md`, cluster runs | `thesis/` |
| **Writing agent** | `thesis/` (drafts, figures chosen for the text, bibliography) | `src/`, `configs/`, `scripts/`, `tests/` |

## Shared source of truth

- **`docs/status.md`**: current decisions, results, pending runs and open questions. The programming agent updates it after every milestone. The writing agent reads it before drafting and never states a result that is not in it or in a results file.
- **Results files:** `runs/<name>/eval_*/metrics.md` and `metrics.json` (test set, 95% bootstrap CIs), `runs/<name>/log.jsonl` (training and validation curves), `runs/<name>/previews/`.
- **Background:** `README.md` (sampling theory, operators), `docs/literature_review.md` (references; entries marked [?] still need verification), `docs/data_findings.md` (data diagnostics), `docs/apolo.md` (cluster).
- The writing agent asks the programming agent (or the user) when a number is missing or unclear, and never invents or extrapolates results.

## Confidentiality

Some references are internal and unpublished. They are listed in the Claude auto-memory for this project, not in the repo. Never copy their content into any file in this repo (it syncs to GitHub) or into any external service. Cite them only as the user instructs.

## Conventions

- Real tomograms only (`data.sources: [phase]`), K=2 lateral decimation, split by sample (`configs/split.yaml`).
- Phase metrics are computed only where the signal is ≥ 10 dB above the B-scan noise floor.
- Commits: `Type(scope): Summary.`, with the Claude co-author trailer.
