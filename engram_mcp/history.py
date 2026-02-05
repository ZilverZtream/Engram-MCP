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
        # Validate that repo_path is a git repository
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

        # Parse git log
        commits = await self._parse_git_log(safe_repo, limit=limit, branch=branch)
        logging.info("Parsed %d commits from git log", len(commits))

        # Filter meaningful commits
        meaningful_commits = self._filter_meaningful_commits(commits)
        logging.info(
            "Filtered to %d meaningful commits (%.1f%%)",
            len(meaningful_commits),
            100.0 * len(meaningful_commits) / max(1, len(commits)),
        )

        # Embed commit messages
        embedded_commits = await self._embed_commit_messages(meaningful_commits)
        logging.info("Embedded %d commit messages", len(embedded_commits))

        # Parse diffs for each commit
        all_diffs: List[GitDiff] = []
        all_tags: List[Tuple[str, str]] = []
        for commit in meaningful_commits:
            diffs = await self._parse_commit_diffs(safe_repo, commit.commit_hash)
            all_diffs.extend(diffs)

            # Auto-tag commits based on message
            tags = self._extract_commit_tags(commit.message)
            for tag in tags:
                all_tags.append((commit.commit_hash, tag))

        logging.info("Parsed %d diffs across all commits", len(all_diffs))
        logging.info("Extracted %d tags", len(all_tags))

        # Store in database
        await self._store_commits(project_id, embedded_commits)
        await self._store_diffs(all_diffs)
        await self._store_tags(all_tags)

        return {
            "commits": len(embedded_commits),
            "diffs": len(all_diffs),
            "tags": len(all_tags),
        }

    async def _parse_git_log(
        self,
        repo_path: str,
        *,
        limit: int,
        branch: str,
    ) -> List[GitCommit]:
        """Parse git log using subprocess."""
        # Format: hash|author|timestamp|subject
        # We use --name-only to get changed files
        cmd = [
            "git",
            "-C",
            repo_path,
            "log",
            branch,
            f"--max-count={limit}",
            "--format=%H|%an|%at|%s",
            "--name-only",
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

    def _parse_log_output(self, output: str) -> List[GitCommit]:
        """Parse git log output into GitCommit objects."""
        commits: List[GitCommit] = []
        lines = output.strip().split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Parse commit header
            if "|" in line:
                parts = line.split("|", 3)
                if len(parts) < 4:
                    i += 1
                    continue

                commit_hash, author, timestamp_str, message = parts
                try:
                    timestamp = int(timestamp_str)
                except ValueError:
                    i += 1
                    continue

                # Collect changed files (until next commit or end)
                changed_files: List[str] = []
                i += 1
                while i < len(lines):
                    file_line = lines[i].strip()
                    if not file_line:
                        i += 1
                        break
                    # Next commit starts
                    if "|" in file_line:
                        break
                    changed_files.append(file_line)
                    i += 1

                commits.append(
                    GitCommit(
                        commit_hash=commit_hash,
                        author=author,
                        timestamp=timestamp,
                        message=message,
                        changed_files=changed_files,
                    )
                )
            else:
                i += 1

        return commits

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

    async def _parse_commit_diffs(
        self, repo_path: str, commit_hash: str
    ) -> List[GitDiff]:
        """Parse diffs for a specific commit."""
        cmd = [
            "git",
            "-C",
            repo_path,
            "show",
            "--format=",
            "--name-status",
            commit_hash,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logging.warning("Failed to get diff for commit %s: %s", commit_hash, stderr)
            return []

        diffs = []
        output = stdout.decode("utf-8", errors="replace")

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Format: STATUS\tFILENAME
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            status, file_path = parts

            # Map git status to our change types
            change_type = "MODIFIED"
            if status.startswith("A"):
                change_type = "ADDED"
            elif status.startswith("D"):
                change_type = "DELETED"
            elif status.startswith("M"):
                change_type = "MODIFIED"

            # Get the actual diff content for this file
            diff_content = await self._get_file_diff(
                repo_path, commit_hash, file_path
            )

            # Create deterministic ID
            diff_id = hashlib.sha256(
                f"{commit_hash}:{file_path}".encode("utf-8")
            ).hexdigest()

            diffs.append(
                GitDiff(
                    commit_hash=commit_hash,
                    file_path=file_path,
                    change_type=change_type,
                    diff_content=diff_content,
                )
            )

        return diffs

    async def _get_file_diff(
        self, repo_path: str, commit_hash: str, file_path: str
    ) -> str:
        """Get the actual diff content for a specific file in a commit."""
        cmd = [
            "git",
            "-C",
            repo_path,
            "show",
            f"{commit_hash}:{file_path}",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # File might have been deleted or renamed
            return ""

        # Truncate large diffs to first 2000 chars
        content = stdout.decode("utf-8", errors="replace")
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"

        return content

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

    async def _store_commits(
        self,
        project_id: str,
        commits: List[Tuple[GitCommit, str, np.ndarray]],
    ) -> None:
        """Store commits in database."""
        for commit, uuid, _embedding in commits:
            await dbmod.upsert_git_commit(
                self.db_path,
                project_id=project_id,
                commit_hash=commit.commit_hash,
                author=commit.author,
                timestamp=commit.timestamp,
                message=commit.message,
                embedding_uuid=uuid,
            )

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
