#!/usr/bin/env python3
"""
Auto-generates README.md index from a folder-based note structure.
Each top-level folder = one topic. Numbered files within = subtopics in order.
Run locally or via GitHub Actions on every push.
"""

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXCLUDE_DIRS = {"scripts", ".github", ".git"}
EXCLUDE_ROOT_FILES = {"README.md", "CHANGELOG.md"}
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
README_PATH = REPO_ROOT / "README.md"


def get_title(md_file: Path) -> str:
    """Return the first # heading from a file, or fall back to filename stem."""
    with md_file.open(encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^#\s+(.+)", line.rstrip())
            if m:
                return m.group(1).strip()
    return md_file.stem


def get_subheadings(md_file: Path) -> list[str]:
    """Return all ## headings from a file."""
    headings = []
    with md_file.open(encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^##\s+(.+)", line.rstrip())
            if m:
                headings.append(m.group(1).strip())
    return headings


def url_encode(path: str) -> str:
    return path.replace(" ", "%20")


def discover_topics() -> list[tuple[str, list[Path]]]:
    """
    Returns list of (topic_folder_name, sorted_md_files).
    Only top-level folders are topics. Files directly in root are ignored.
    """
    topics = []
    for item in sorted(REPO_ROOT.iterdir()):
        if not item.is_dir():
            continue
        if item.name in EXCLUDE_DIRS or item.name.startswith("."):
            continue
        md_files = sorted(item.glob("*.md"))
        if md_files:
            topics.append((item.name, md_files))
    return topics


def get_changed_files() -> list[str]:
    """Return .md files changed in the last commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        changed = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line.endswith(".md"):
                continue
            if Path(line).name in EXCLUDE_ROOT_FILES:
                continue
            parts = Path(line).parts
            if any(p in EXCLUDE_DIRS for p in parts):
                continue
            changed.append(line)
        return changed
    except Exception:
        return []


def read_changelog_rows() -> list[str]:
    if not CHANGELOG_PATH.exists():
        return []
    rows = []
    in_table = False
    with CHANGELOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("| Date"):
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if in_table and line.startswith("|"):
                rows.append(line)
    return rows


def update_changelog(changed_files: list[str]) -> list[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new_rows = []
    for filepath in changed_files:
        md_file = REPO_ROOT / filepath
        title = get_title(md_file) if md_file.exists() else Path(filepath).stem
        topic = Path(filepath).parts[0] if len(Path(filepath).parts) > 1 else "root"
        new_rows.append(f"| {now} | {topic} | {title} | Added/Updated |")

    existing_rows = read_changelog_rows()
    all_rows = new_rows + existing_rows  # newest first

    header = "# Changelog\n\n| Date (UTC) | Topic | Subtopic | Action |\n|---|---|---|---|\n"
    with CHANGELOG_PATH.open("w", encoding="utf-8") as f:
        f.write(header)
        for row in all_rows:
            f.write(row + "\n")

    return all_rows


def build_readme(topics: list[tuple[str, list[Path]]], changelog_rows: list[str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_files = sum(len(files) for _, files in topics)

    topic_sections = []
    for folder_name, md_files in topics:
        display_name = folder_name.replace("-", " ").replace("_", " ")
        rows = []
        for i, md_file in enumerate(md_files, 1):
            title = get_title(md_file)
            subheadings = get_subheadings(md_file)
            rel_path = url_encode(str(md_file.relative_to(REPO_ROOT)))
            sub_preview = ", ".join(subheadings[:3])
            if len(subheadings) > 3:
                sub_preview += f" +{len(subheadings) - 3} more"
            sub_col = f"_{sub_preview}_" if sub_preview else "—"
            rows.append(f"| {i} | [{title}]({rel_path}) | {sub_col} |")

        table = f"### {display_name}\n\n| # | Subtopic | Covers |\n|---|----------|--------|\n"
        table += "\n".join(rows)
        topic_sections.append(table)

    toc = "\n\n".join(topic_sections) if topic_sections else "_No topics found._"

    # Recent updates: last 10 rows
    recent = changelog_rows[:10]
    if recent:
        updates = "| Date (UTC) | Topic | Subtopic | Action |\n|---|---|---|---|\n"
        updates += "\n".join(recent)
    else:
        updates = "_No updates recorded yet._"

    return f"""# AI Learning Repository

> Auto-generated index. Last updated: {now}
> {len(topics)} topic(s) · {total_files} subtopic file(s)

---

## Topics

{toc}

---

## Learning Log

{updates}

---

_Auto-generated by `scripts/update_readme.py`. Do not edit by hand._
_To add a topic: create a new folder. To add a subtopic: add a numbered `.md` file inside it._
"""


def main():
    topics = discover_topics()
    changed_files = get_changed_files()

    print(f"Found {len(topics)} topic folder(s).")
    print(f"Changed in last commit: {changed_files or 'none (local run)'}")

    # Local run with no git diff: treat all files as new
    if not changed_files:
        changed_files = [
            str(f.relative_to(REPO_ROOT))
            for _, files in topics
            for f in files
        ]

    changelog_rows = update_changelog(changed_files)
    readme_content = build_readme(topics, changelog_rows)

    README_PATH.write_text(readme_content, encoding="utf-8")
    print(f"README.md written.")
    print(f"CHANGELOG.md updated ({len(changelog_rows)} total rows).")


if __name__ == "__main__":
    main()
