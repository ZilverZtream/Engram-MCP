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
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from . import db as dbmod
from .embeddings import EmbeddingService
from .security import PathContext

_COMMIT_MARKER = "===COMMIT_START==="
_MAX_DIFF_CHARS = 2000
_MAX_COMMIT_SECTION_BYTES = 1_048_576  # 1MB safety valve per commit


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

        Uses streaming parsing to avoid loading the entire output into RAM.
        The custom ``--format`` string embeds a unique delimiter before each
        commit's metadata line. Everything between two delimiters — metadata,
        name-status, and the full unified patch — belongs to the same commit.

        Memory usage: O(single commit size) instead of O(total log size).
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

        commits: List[GitCommit] = []
        all_diffs: List[GitDiff] = []

        async for commit, diffs in self._stream_parse_git_log(proc.stdout):
            commits.append(commit)
            all_diffs.extend(diffs)

        await proc.wait()

        if proc.returncode != 0:
            stderr_data = await proc.stderr.read()
            error_msg = stderr_data.decode("utf-8", errors="replace")
            raise RuntimeError(f"git log failed: {error_msg}")

        return commits, all_diffs

    async def _stream_parse_git_log(
        self,
        stdout: asyncio.StreamReader,
    ) -> AsyncIterator[Tuple[GitCommit, List[GitDiff]]]:
        """Stream-parse git log output line-by-line, yielding commits incrementally.

        Accumulates lines for each commit section, then offloads the CPU-intensive
        parsing to a thread pool. This prevents:
        1. Memory spike: only one commit section in RAM at a time
        2. Event loop blocking: heavy parsing happens in a thread

        Implements a 1MB safety valve: if a commit section exceeds the limit,
        it truncates and yields a warning in the diff content.
        """
        loop = asyncio.get_running_loop()
        commit_section_lines: List[str] = []
        commit_section_bytes = 0
        truncated = False

        async for line_bytes in stdout:
            line = line_bytes.decode("utf-8", errors="replace")

            if line.startswith(_COMMIT_MARKER):
                if commit_section_lines:
                    section_text = "".join(commit_section_lines)

                    commit, diffs = await loop.run_in_executor(
                        None, self._parse_single_commit_section, section_text, truncated
                    )

                    if commit is not None:
                        yield commit, diffs

                    commit_section_lines = [line]
                    commit_section_bytes = len(line_bytes)
                    truncated = False
                else:
                    commit_section_lines.append(line)
                    commit_section_bytes += len(line_bytes)
            else:
                if commit_section_bytes + len(line_bytes) > _MAX_COMMIT_SECTION_BYTES:
                    if not truncated:
                        logging.warning(
                            "Commit section exceeds 1MB limit, truncating to prevent memory spike"
                        )
                        commit_section_lines.append(
                            "\n... (commit section truncated due to size limit)\n"
                        )
                        truncated = True
                else:
                    commit_section_lines.append(line)
                    commit_section_bytes += len(line_bytes)

        if commit_section_lines:
            section_text = "".join(commit_section_lines)

            commit, diffs = await loop.run_in_executor(
                None, self._parse_single_commit_section, section_text, truncated
            )

            if commit is not None:
                yield commit, diffs

    def _parse_single_commit_section(
        self, section: str, truncated: bool = False
    ) -> Tuple[Optional[GitCommit], List[GitDiff]]:
        """Parse a single commit section (runs in thread pool to avoid blocking event loop).

        Args:
            section: The raw text for one commit (everything between two markers)
            truncated: Whether this section was truncated due to size limits

        Returns:
            (GitCommit, List[GitDiff]) or (None, []) if parsing fails
        """
        if not section.strip():
            return None, []

        newline_pos = section.find("\n")
        if newline_pos < 0:
            return None, []

        header = section[:newline_pos].strip()
        body = section[newline_pos + 1:]

        if header.startswith(_COMMIT_MARKER):
            header = header[len(_COMMIT_MARKER):]

        parts = header.split("|", 3)
        if len(parts) < 4:
            return None, []

        commit_hash, author, timestamp_str, message = parts
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            return None, []

        diff_start = body.find("diff --git ")
        if diff_start >= 0:
            name_status_text = body[:diff_start]
            diff_text = body[diff_start:]
        else:
            name_status_text = body
            diff_text = ""

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

        commit = GitCommit(
            commit_hash=commit_hash,
            author=author,
            timestamp=timestamp,
            message=message,
            changed_files=changed_files,
        )

        all_diffs: List[GitDiff] = []
        if diff_text:
            for block in re.split(r"(?=diff --git )", diff_text):
                if not block.strip():
                    continue

                file_path = self._extract_filepath_from_diff(block)
                if not file_path:
                    continue

                change_type = change_type_map.get(file_path, "MODIFIED")

                if truncated:
                    diff_content = block[:_MAX_DIFF_CHARS] + "\n... (truncated - commit section exceeded 1MB)"
                else:
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

        return commit, all_diffs

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
        """Bulk-store commits and their embeddings as searchable chunks."""
        if not commits:
            return

        # Store commit metadata in git_commits table
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

        # Store embeddings as chunks for vector search
        # This enables semantic history search via FAISS
        chunk_tuples = []
        for commit, uuid, embedding in commits:
            # Create a chunk with the commit message content
            chunk_id = uuid  # Use the embedding_uuid as chunk_id
            # Approximate token count (rough heuristic: ~0.75 tokens per char)
            token_count = max(1, int(len(commit.message) * 0.75))
            metadata = {
                "type": "git_commit",
                "commit_hash": commit.commit_hash,
                "author": commit.author,
                "timestamp": commit.timestamp,
                "file_path": "vfs://git_history",  # Virtual file system path
            }
            import json
            chunk_tuples.append((
                chunk_id,
                project_id,
                commit.message,  # Content is the commit message
                token_count,
                json.dumps(metadata),
                embedding,
            ))

        if chunk_tuples:
            await dbmod.upsert_chunks_with_embeddings(
                self.db_path,
                project_id=project_id,
                chunks=chunk_tuples
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

    # ------------------------------------------------------------------
    # Immune System: Revert Detection
    # ------------------------------------------------------------------

    async def process_reverts(
        self,
        *,
        project_id: str,
    ) -> Dict[str, int]:
        """Detect revert commits and auto-generate repo_rules to prevent regressions.

        This implements "The Immune System" - learning from mistakes by analyzing
        reverted commits and treating their patterns as anti-patterns.

        Returns:
            Dict with counts: {"reverts_found": N, "rules_generated": M}
        """
        # Fetch all commits for this project
        commits = await dbmod.fetch_git_commits(
            self.db_path,
            project_id=project_id,
            limit=1000,  # Process last 1000 commits
        )

        reverts_found = []
        revert_patterns = [
            r"^revert[:\s]+[\"\']?(.+?)[\"\']?$",  # Revert "commit message"
            r"^revert\s+([a-f0-9]{7,40})",  # Revert <hash>
            r"this reverts commit\s+([a-f0-9]{7,40})",  # This reverts commit <hash>
        ]

        for commit in commits:
            message = commit["message"].strip()
            message_lower = message.lower()

            # Check if this is a revert commit
            is_revert = False
            reverted_hash = None
            original_message = None

            for pattern in revert_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    is_revert = True
                    matched_text = match.group(1)

                    # Try to extract hash (7-40 hex chars)
                    hash_match = re.search(r"([a-f0-9]{7,40})", matched_text)
                    if hash_match:
                        reverted_hash = hash_match.group(1)
                    else:
                        # The matched text is the original message
                        original_message = matched_text

                    break

            if not is_revert:
                continue

            # Find the original (bad) commit
            original_commit = None
            if reverted_hash:
                # Try exact hash match
                for c in commits:
                    if c["commit_hash"].startswith(reverted_hash):
                        original_commit = c
                        break

            if not original_commit and original_message:
                # Try message match
                for c in commits:
                    if original_message in c["message"].lower():
                        original_commit = c
                        break

            if original_commit:
                reverts_found.append({
                    "revert_commit_hash": commit["commit_hash"],
                    "revert_message": message,
                    "original_commit_hash": original_commit["commit_hash"],
                    "original_message": original_commit["message"],
                })

        # Generate rules for each revert
        rules_generated = 0
        for revert_info in reverts_found:
            original_hash = revert_info["original_commit_hash"]
            original_message = revert_info["original_message"]
            revert_message = revert_info["revert_message"]

            # Fetch diffs from the original (bad) commit
            diffs = await dbmod.fetch_git_diffs_for_commit(
                self.db_path,
                commit_hash=original_hash,
            )

            if not diffs:
                continue

            # Extract file patterns from the bad commit
            affected_files = [d["file_path"] for d in diffs]
            file_pattern = "|".join(affected_files[:5])  # Limit to 5 files

            # Generate rule text
            rule_text = (
                f"CRITICAL: Anti-pattern detected from reverted commit {original_hash[:8]}.\n\n"
                f"Original commit: '{original_message}'\n"
                f"This commit was REVERTED because it caused a regression.\n\n"
                f"Affected files: {', '.join(affected_files[:3])}\n"
                f"{'...' if len(affected_files) > 3 else ''}\n\n"
                f"Before making similar changes, review why this was reverted:\n"
                f"Revert message: '{revert_message}'\n\n"
                f"Recommendation: Search git history for this commit to understand the failure."
            )

            # Create rule ID
            rule_id = f"immune_system_{original_hash[:12]}"

            # Insert into repo_rules
            await dbmod.upsert_repo_rule(
                self.db_path,
                rule_id=rule_id,
                project_id=project_id,
                file_pattern=file_pattern,
                rule_text=rule_text,
                priority=10,  # High priority
            )

            rules_generated += 1

        logging.info(
            "Immune system: found %d reverts, generated %d rules for project %s",
            len(reverts_found),
            rules_generated,
            project_id,
        )

        return {
            "reverts_found": len(reverts_found),
            "rules_generated": rules_generated,
        }
