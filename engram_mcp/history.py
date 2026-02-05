"""Episodic Diff Memory: Git History Indexer.

This module enables agents to learn from git history (evolution) rather than
just the current state. It indexes commit messages and diffs to build a
searchable database of cause-and-effect patterns.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import db as dbmod
from .embeddings import EmbeddingService
from .security import PathContext

_COMMIT_MARKER = "===COMMIT_START==="
_MAX_DIFF_CHARS = 2000


@dataclass
class GitCommit:
    """Represents a single git commit with metadata."""

    commit_hash: str
    author: str
    timestamp: int
    message: str
    changed_files: List[str]


@dataclass
class GitDiff:
    """Represents a diff for a single file in a commit."""

    commit_hash: str
    file_path: str
    change_type: str  # 'ADDED', 'MODIFIED', 'DELETED'
    diff_content: str


class GitIndexer:
    """Indexes git history for semantic search of fixes and changes."""

    def __init__(
        self,
        *,
        db_path: str,
        embedding_service: EmbeddingService,
        project_path_context: PathContext,
    ):
        self.db_path = db_path
        self.embedding_service = embedding_service
        self.project_path_context = project_path_context

    async def index_git_history(
        self,
        *,
        project_id: str,
        repo_path: str,
        limit: int = 500,
        branch: str = "HEAD",
    ) -> Dict[str, int]:
        """Index git history for a project.

        Args:
            project_id: The project ID to associate commits with
            repo_path: Path to the git repository
            limit: Maximum number of commits to index
            branch: Git branch/ref to index (default: HEAD)

        Returns:
            Dict with counts of indexed commits, diffs, and tags
        """
        safe_repo = str(self.project_path_context.resolve_path(repo_path))
        git_dir = os.path.join(safe_repo, ".git")
        if not os.path.exists(git_dir):
            raise ValueError(f"Not a git repository: {repo_path}")

        logging.info(
            "Starting git history indexing for project %s (limit=%d, branch=%s)",
            project_id,
            limit,
            branch,
        )

        # Single subprocess retrieves commit metadata AND diffs together.
        commits, all_diffs = await self._parse_git_log(
            safe_repo, limit=limit, branch=branch
        )
        logging.info(
            "Parsed %d commits and %d diffs from git log",
            len(commits),
            len(all_diffs),
        )

        # Filter meaningful commits and keep only their diffs.
        meaningful_commits = self._filter_meaningful_commits(commits)
        meaningful_hashes = {c.commit_hash for c in meaningful_commits}
        filtered_diffs = [d for d in all_diffs if d.commit_hash in meaningful_hashes]
        logging.info(
            "Filtered to %d meaningful commits (%.1f%%), %d diffs",
            len(meaningful_commits),
            100.0 * len(meaningful_commits) / max(1, len(commits)),
            len(filtered_diffs),
        )

        # Extract tags (pure CPU, no I/O).
        all_tags: List[Tuple[str, str]] = [
            (commit.commit_hash, tag)
            for commit in meaningful_commits
            for tag in self._extract_commit_tags(commit.message)
        ]
        logging.info("Extracted %d tags", len(all_tags))

        # Overlap GPU work (embedding) with DB writes (diffs + tags).
        embedded_commits, _, _ = await asyncio.gather(
            self._embed_commit_messages(meaningful_commits),
            self._store_diffs(filtered_diffs),
            self._store_tags(all_tags),
        )
        logging.info("Embedded %d commit messages", len(embedded_commits))

        # Commits stored last — the embedding UUIDs from the gather are needed.
        await self._store_commits(project_id, embedded_commits)

        return {
            "commits": len(embedded_commits),
            "diffs": len(filtered_diffs),
            "tags": len(all_tags),
        }

    # ------------------------------------------------------------------
    # Git log parsing — single subprocess
    # ------------------------------------------------------------------

    async def _parse_git_log(
        self,
        repo_path: str,
        *,
        limit: int,
        branch: str,
    ) -> Tuple[List[GitCommit], List[GitDiff]]:
        """Retrieve commits and diffs in a single ``git log -p`` subprocess.

        The custom ``--format`` string embeds a unique delimiter before each
        commit's metadata line.  Everything between two delimiters — metadata,
        name-status, and the full unified patch — belongs to the same commit.
        This eliminates the O(N*M) subprocess fan-out that the previous
        per-commit ``git show`` approach required.
        """
        cmd = [
            "git",
            "-C",
            repo_path,
            "log",
            branch,
            f"--max-count={limit}",
            f"--format={_COMMIT_MARKER}%H|%an|%at|%s",
            "--name-status",
            "-p",
            "--no-merges",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"git log failed: {error_msg}")

        return self._parse_log_output(stdout.decode("utf-8", errors="replace"))

    def _parse_log_output(self, output: str) -> Tuple[List[GitCommit], List[GitDiff]]:
        """State-machine parser for ``git log -p --name-status`` output.

        Layout per commit::

            ===COMMIT_START===<hash>|<author>|<ts>|<subject>
            <status>\\t<path>        ← name-status block
            ...
            diff --git a/<path> b/<path>   ← patch blocks
            <diff header + body>
            ...

        The parser splits on the commit marker, then within each section
        splits on the first ``diff --git`` occurrence to separate
        name-status metadata from patch content.
        """
        commits: List[GitCommit] = []
        all_diffs: List[GitDiff] = []

        for section in output.split(_COMMIT_MARKER)[1:]:
            if not section.strip():
                continue

            # --- Header (first line) ---------------------------------------------
            newline_pos = section.find("\n")
            if newline_pos < 0:
                continue
            header = section[:newline_pos].strip()
            body = section[newline_pos + 1:]

            parts = header.split("|", 3)
            if len(parts) < 4:
                continue
            commit_hash, author, timestamp_str, message = parts
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                continue

            # --- Split body into name-status vs patch sections -------------------
            diff_start = body.find("diff --git ")
            if diff_start >= 0:
                name_status_text = body[:diff_start]
                diff_text = body[diff_start:]
            else:
                name_status_text = body
                diff_text = ""

            # --- Parse name-status → {file_path: change_type} -------------------
            change_type_map: Dict[str, str] = {}
            changed_files: List[str] = []
            for line in name_status_text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                tab_parts = stripped.split("\t")
                if len(tab_parts) < 2:
                    continue
                status = tab_parts[0]
                # Rename/Copy: R###\told\tnew  or  C###\told\tnew
                if (status.startswith("R") or status.startswith("C")) and len(tab_parts) >= 3:
                    file_path = tab_parts[2]
                else:
                    file_path = tab_parts[1]

                if status.startswith("A"):
                    change_type_map[file_path] = "ADDED"
                elif status.startswith("D"):
                    change_type_map[file_path] = "DELETED"
                else:
                    change_type_map[file_path] = "MODIFIED"
                changed_files.append(file_path)

            commits.append(
                GitCommit(
                    commit_hash=commit_hash,
                    author=author,
                    timestamp=timestamp,
                    message=message,
                    changed_files=changed_files,
                )
            )

            # --- Parse per-file diffs --------------------------------------------
            if not diff_text:
                continue
            for block in re.split(r"(?=diff --git )", diff_text):
                if not block.strip():
                    continue

                file_path = self._extract_filepath_from_diff(block)
                if not file_path:
                    continue

                change_type = change_type_map.get(file_path, "MODIFIED")
                diff_content = (
                    block
                    if len(block) <= _MAX_DIFF_CHARS
                    else block[:_MAX_DIFF_CHARS] + "\n... (truncated)"
                )

                all_diffs.append(
                    GitDiff(
                        commit_hash=commit_hash,
                        file_path=file_path,
                        change_type=change_type,
                        diff_content=diff_content,
                    )
                )

        return commits, all_diffs

    @staticmethod
    def _extract_filepath_from_diff(block: str) -> Optional[str]:
        """Extract the canonical file path from a single diff block.

        Checks ``+++ b/<path>`` first (present for all non-deleted files),
        falls back to ``--- a/<path>`` for deleted files, and handles the
        ``Binary files … differ`` line emitted for binary diffs.
        """
        minus_path: Optional[str] = None
        for line in block.split("\n")[:15]:  # header is always in the first few lines
            if line.startswith("+++ b/"):
                return line[6:]
            if line.startswith("--- a/") and minus_path is None:
                minus_path = line[6:]
            # Binary file: no +++ / --- lines
            bin_match = re.match(r"^Binary files? a/.+ and b/(.+) differ$", line)
            if bin_match:
                return bin_match.group(1)
        return minus_path

    # ------------------------------------------------------------------
    # Filtering & tagging
    # ------------------------------------------------------------------

    def _filter_meaningful_commits(self, commits: List[GitCommit]) -> List[GitCommit]:
        """Filter out noise commits (version bumps, merges, etc.)."""
        meaningful = []
        skip_patterns = [
            r"^bump version",
            r"^merge branch",
            r"^merge pull request",
            r"^wip$",
            r"^update changelog",
            r"^version \d",
        ]

        for commit in commits:
            msg_lower = commit.message.lower().strip()

            # Skip if message is too short
            if len(msg_lower) < 10:
                continue

            # Skip if matches a noise pattern
            if any(re.search(pattern, msg_lower) for pattern in skip_patterns):
                continue

            meaningful.append(commit)

        return meaningful

    def _extract_commit_tags(self, message: str) -> List[str]:
        """Extract semantic tags from commit message."""
        tags = []
        msg_lower = message.lower()

        # Common patterns
        patterns = {
            "fix": [r"\bfix\b", r"\bfixes\b", r"\bfixed\b", r"\bbugfix\b"],
            "refactor": [r"\brefactor\b", r"\brefactoring\b"],
            "perf": [r"\bperf\b", r"\bperformance\b", r"\boptimize\b"],
            "feat": [r"\bfeat\b", r"\bfeature\b", r"\badd\b"],
            "docs": [r"\bdocs\b", r"\bdocumentation\b"],
            "test": [r"\btest\b", r"\btests\b", r"\btesting\b"],
            "style": [r"\bstyle\b", r"\bformatting\b"],
            "chore": [r"\bchore\b"],
        }

        for tag, regexes in patterns.items():
            if any(re.search(regex, msg_lower) for regex in regexes):
                tags.append(tag)

        return tags

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed_commit_messages(
        self, commits: List[GitCommit]
    ) -> List[Tuple[GitCommit, str, np.ndarray]]:
        """Embed commit messages and return (commit, uuid, embedding) tuples."""
        if not commits:
            return []

        messages = [commit.message for commit in commits]

        # Embed all messages in batch
        embeddings = await self.embedding_service.embed_batch(messages)

        # Generate UUIDs for embeddings (for future FAISS indexing)
        result = []
        for commit, embedding in zip(commits, embeddings):
            # Generate a deterministic UUID based on project + commit hash
            uuid = hashlib.sha256(
                f"{commit.commit_hash}:{commit.message}".encode("utf-8")
            ).hexdigest()[:32]
            result.append((commit, uuid, embedding))

        return result

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    async def _store_commits(
        self,
        project_id: str,
        commits: List[Tuple[GitCommit, str, np.ndarray]],
    ) -> None:
        """Bulk-store commits via a single executemany round-trip."""
        if not commits:
            return
        commit_tuples = [
            (
                commit.commit_hash,
                project_id,
                commit.author,
                commit.timestamp,
                commit.message,
                uuid,
            )
            for commit, uuid, _embedding in commits
        ]
        await dbmod.upsert_git_commits(self.db_path, commits=commit_tuples)

    async def _store_diffs(self, diffs: List[GitDiff]) -> None:
        """Store diffs in database."""
        if not diffs:
            return

        diff_tuples = []
        for diff in diffs:
            diff_id = hashlib.sha256(
                f"{diff.commit_hash}:{diff.file_path}".encode("utf-8")
            ).hexdigest()
            diff_tuples.append(
                (
                    diff_id,
                    diff.commit_hash,
                    diff.file_path,
                    diff.change_type,
                    diff.diff_content,
                )
            )

        await dbmod.upsert_git_diffs(self.db_path, diffs=diff_tuples)

    async def _store_tags(self, tags: List[Tuple[str, str]]) -> None:
        """Store tags in database."""
        if not tags:
            return
        await dbmod.upsert_git_tags(self.db_path, tags=tags)
