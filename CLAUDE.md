# AI Learning Repository — Claude Context

## Purpose
Personal AI/ML learning journal for a software engineer.
Goals:
- Accountability: every topic read is tracked with a UTC timestamp in CHANGELOG.md
- Quick retrieval: for interview prep or brushing up on a specific subtopic without hunting through browser bookmarks
- Progressive depth: files are numbered so chapter 1 = overview, later chapters = application and implementation

## Repository Structure
- Each top-level folder = one AI/ML topic (e.g. `AI-Infrastructure/`, `Transformers/`)
- Each numbered `.md` file inside = one subtopic in reading order
  - `01-Overview.md` = superficial intro
  - `02-...` through `0N-...` = progressively deeper
- `README.md` — AUTO-GENERATED. Never edit by hand.
- `CHANGELOG.md` — AUTO-GENERATED. Never edit by hand.
- `scripts/update_readme.py` — regenerates README + CHANGELOG from current files

## How to Add a New Topic
1. Create a new folder: `mkdir TopicName/`
2. Add numbered files: `01-Overview.md`, `02-DeepDive.md`, etc.
3. Each file: first line must be a `#` heading (used as the subtopic title in README)
4. Use `##` subheadings freely — they appear as "Covers" preview in README
5. Run: `python3 scripts/update_readme.py`
6. Commit: `git add . && git commit -m "feat: add [Topic] notes"`

## How to Add a Subtopic to an Existing Topic
1. Add a new numbered file to the topic folder (e.g. `07-NewSubtopic.md`)
2. Run: `python3 scripts/update_readme.py`
3. Commit

## Naming Convention
- Folder names: `PascalCase` or `Hyphenated-Title` (e.g. `AI-Infrastructure`, `Transformers`)
- File names: `NN-Descriptive-Name.md` where NN is zero-padded number (01, 02, ... 10, 11)

## Note Format
- First line: `# Title` (becomes the subtopic link text in README)
- Subheadings `##` and `###` are extracted for the "Covers" preview column
- Use code blocks, tables, diagrams freely — these are personal revision notes
- No need for audience-friendly prose — write for yourself

## What NOT to Do
- Do NOT edit `README.md` or `CHANGELOG.md` directly — they are overwritten on next script run
- Do NOT create `.md` files in the repo root (they will be ignored by the script)
- Do NOT rename folders without re-running the script

## Existing Topics
- `AI-Infrastructure/` — GPU basics, model loading, RunPod, resource optimization, hybrid training, framework stack
