# Implementation Plan — Post-M1 Cleanup & Upgrades

**Source:** `Next_build_go_status.md` feedback from 2026-06-16
**Status:** Planned

## Overview

Four tasks derived from user feedback after Milestone 1 (credit_scoring pull & rewrite).

## Task A — Restructure superclean root

Consolidate the superclean project folder to a minimal agent-navigable structure.

**Current layout:**
```
superclean/
├── .editorconfig, .gitattributes, .gitignore  # git/IDE config
├── .opencode/                                  # agent configs
├── AGENTS.md, Next_build.md, opencode.jsonc    # project files
├── credit_scoring/                             # pulled repo (v2)
├── docs/
│   ├── guide/                                  # 1 file (how-to guide)
│   ├── plans/                                  # M1+M2 plans
│   └── README.md
├── scripts/                                    # pull-and-clean.py
├── src/                                        # placeholder README
├── task-completed/                             # session logs
└── tests/                                      # placeholder README
```

**Target layout:**
```
superclean/
├── .opencode/
├── AGENTS.md, Next_build.md, opencode.jsonc
├── credit_scoring/
├── docs/
│   ├── plans/
│   ├── complete_repo_inventory.md
│   ├── free-hosting-options.md
│   └── README.md
├── scripts/
├── src/
├── tests/
└── task-completed/
```

**Actions:**
1. Flatten `docs/guide/defining-custom-subagents.md` → `docs/defining-custom-subagents.md`
2. Remove empty `docs/guide/` directory
3. Update `AGENTS.md` cross-references to match new paths
4. Update `Next_build.md` if needed

## Task B — Write complete_repo_inventory.md

Create `docs/complete_repo_inventory.md` enumerating every repo pulled into superclean.

| Repo | Status | Purpose | Key outputs |
|------|--------|---------|-------------|
| credit_scoring | Cleaned ✓ | Credit scoring ML pipeline (ingest → serving) | 14 source files, 6 test files, pyproject.toml |

Include: purpose, status (cleaned/pending), dependency/relationship map.

## Task C — Research free hosting alternatives

Investigate free-tier hosting for Python/Streamlit/FastAPI.

**Platforms:** Render, Fly.io, Railway, Cloudflare Pages, Vercel, PythonAnywhere

**Criteria:** free tier limits, cold-start latency, Python support, Streamlit/FastAPI compatibility.

**Deliverable:** `docs/free-hosting-options.md` with comparison table and recommendation.

## Task D — credit_scoring/upgrades deliverable

Create `credit_scoring/upgrades/` with a self-contained cleaning process analysis.

**Deliverable 1:** `upgrades/cleaning_report.py`
- Loads before/after metrics (SLOC, complexity, test count, hardcoded paths)
- Generates structured comparison
- Outputs markdown report

**Deliverable 2:** `upgrades/cleaning_report_credit_scoring.md`
- Was the original code clean? (baseline assessment)
- What specific changes were made and why? (catalogue of interventions)
- Runtime vs output quality impact (before/after metrics)

## Execution Order

1. **Task A** — Foundation (paths must be correct before docs)
2. **Task B** — Quick doc from existing exploration data
3. **Task D** — Main deliverable (script + report)
4. **Task C** — Independent research (can be delegated)
