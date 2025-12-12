"""
GitHub connector for repository documentation ingestion.
Supports polling for changes and webhook-based real-time updates.
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from src.connectors.base import BaseConnector, IngestionEvent

logger = logging.getLogger(__name__)


class GitHubConnector(BaseConnector):
    """
    GitHub connector for ingesting markdown/docs from repositories.
    Polls for file changes and processes push/PR webhooks.
    """

    def __init__(self, config, queue, circuit_breaker=None):
        super().__init__(config, queue, circuit_breaker)

        self.token = os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "wekadocs-mcp-connector",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

        # Extract repo from config metadata
        self.owner = config.metadata.get("owner")
        self.repo = config.metadata.get("repo")
        self.docs_path = config.metadata.get("docs_path", "docs")

        if not self.owner or not self.repo:
            raise ValueError("GitHub connector requires 'owner' and 'repo' in metadata")

        logger.info(
            f"GitHub connector initialized: {self.owner}/{self.repo}, "
            f"docs_path={self.docs_path}"
        )

    @staticmethod
    def _map_status_to_event(status: str) -> Optional[str]:
        if status == "added":
            return "created"
        if status == "modified":
            return "updated"
        if status == "removed":
            return "deleted"
        return None

    def _is_docs_file(self, filename: str) -> bool:
        return filename.startswith(self.docs_path) and filename.endswith(
            (".md", ".markdown", ".html")
        )

    def _create_event(
        self,
        *,
        commit_sha: str,
        filename: str,
        event_type: str,
        commit_message: str,
        author: str,
        timestamp: datetime,
        webhook: bool = False,
    ) -> IngestionEvent:
        metadata = {
            "owner": self.owner,
            "repo": self.repo,
            "filename": filename,
            "commit_sha": commit_sha,
            "commit_message": commit_message,
            "author": author,
        }
        if webhook:
            metadata["webhook"] = True

        return IngestionEvent(
            source_uri=f"https://github.com/{self.owner}/{self.repo}/blob/{commit_sha}/{filename}",
            source_type="github",
            event_type=event_type,
            metadata=metadata,
            timestamp=timestamp,
        )

    async def fetch_changes(
        self, since: Optional[str] = None
    ) -> tuple[List[IngestionEvent], Optional[str]]:
        """
        Fetch documentation file changes from GitHub repository.
        Uses commits API to detect changes since last cursor (commit SHA).

        Args:
            since: Commit SHA to start from (None = latest commit only)

        Returns:
            (events, next_cursor) where next_cursor is latest commit SHA
        """
        async with httpx.AsyncClient() as client:
            try:
                commits_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits"
                since_iso, since_sha = self._parse_cursor(since)
                params = {"per_page": self.config.batch_size}
                if since_iso:
                    params["since"] = since_iso

                commits: List[Dict[str, Any]] = []
                next_url: Optional[str] = commits_url
                next_params = params

                while next_url:
                    response = await client.get(
                        next_url,
                        headers=self.headers,
                        params=next_params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    batch = response.json()
                    if not batch:
                        break
                    commits.extend(batch)

                    if since_sha and any(c.get("sha") == since_sha for c in batch):
                        break

                    link_next = response.links.get("next")
                    if link_next and link_next.get("url"):
                        next_url = link_next["url"]
                        next_params = None
                    else:
                        break

                if since_sha and commits:
                    trimmed: List[Dict[str, Any]] = []
                    for commit in commits:
                        if commit.get("sha") == since_sha:
                            break
                        trimmed.append(commit)
                    commits = trimmed

                if not commits:
                    logger.debug("No new commits found")
                    return [], since

                events = []
                latest_sha = commits[0]["sha"] if commits else since
                latest_timestamp = (
                    commits[0].get("commit", {}).get("author", {}).get("date")
                    if commits
                    else None
                )

                # Check each commit for docs changes
                for commit in commits:
                    events.extend(await self._events_from_commit(client, commit))

                logger.info(
                    f"Fetched {len(events)} documentation changes from "
                    f"{len(commits)} commits"
                )
                next_cursor = self._serialize_cursor(latest_timestamp, latest_sha)
                return events, next_cursor

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"GitHub API error: {e.response.status_code} - {e.response.text}"
                )
                raise
            except Exception as e:
                logger.error(f"Error fetching changes from GitHub: {e}")
                raise

    async def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GitHub webhook signature using HMAC-SHA256.

        Args:
            payload: Raw webhook payload bytes
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning(
                "Webhook secret not configured, skipping signature verification"
            )
            return True

        # GitHub sends signature as 'sha256=<hex>'
        if not signature.startswith("sha256="):
            logger.warning(f"Invalid signature format: {signature}")
            return False

        expected_sig = signature.split("=")[1]

        # Compute HMAC
        secret = self.config.webhook_secret.encode()
        computed = hmac.new(secret, payload, hashlib.sha256).hexdigest()

        # Constant-time comparison
        return hmac.compare_digest(computed, expected_sig)

    async def webhook_to_event(self, payload: Dict[str, Any]) -> List[IngestionEvent]:
        """
        Convert GitHub webhook payload to IngestionEvent.

        Processes 'push' events to detect docs changes.

        Args:
            payload: GitHub webhook payload

        Returns:
            IngestionEvent or None if no action needed
        """
        try:
            if "commits" not in payload:
                logger.debug("Ignoring non-push webhook event")
                return []

            # Extract commit info
            commits = payload.get("commits", [])
            if not commits:
                return None

            events: List[IngestionEvent] = []
            for commit in commits:
                commit_sha = commit.get("id") or commit.get("sha")
                if not commit_sha:
                    continue

                added = commit.get("added", []) or []
                modified = commit.get("modified", []) or []
                removed = commit.get("removed", []) or []
                all_files = added + modified + removed

                docs_files = [f for f in all_files if self._is_docs_file(f)]
                if not docs_files:
                    continue

                commit_ts_str = commit.get("timestamp")
                commit_ts = (
                    datetime.fromisoformat(commit_ts_str.replace("Z", "+00:00"))
                    if commit_ts_str
                    else datetime.utcnow()
                )
                commit_message = commit.get("message", "")
                author = commit.get("author", {}).get("username") or commit.get(
                    "author", {}
                ).get("name", "unknown")

                for filename in docs_files:
                    if filename in removed:
                        event_type = "deleted"
                    elif filename in added:
                        event_type = "created"
                    else:
                        event_type = "updated"

                    event = self._create_event(
                        commit_sha=commit_sha,
                        filename=filename,
                        event_type=event_type,
                        commit_message=commit_message,
                        author=author,
                        timestamp=commit_ts,
                        webhook=True,
                    )
                    events.append(event)

            logger.info(
                "Webhook events created for %d documentation files across %d commits",
                len(events),
                len(commits),
            )
            return events

        except Exception as e:
            logger.error(
                f"Error converting webhook payload to event: {e}",
                exc_info=True,
            )
            return []

    def _parse_cursor(
        self, cursor: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        if not cursor:
            return None, None
        try:
            data = json.loads(cursor)
            return data.get("since"), data.get("sha")
        except Exception:
            # Backward compatibility: cursor may be raw ISO timestamp or SHA
            if "T" in cursor:
                return cursor, None
            return None, cursor

    def _serialize_cursor(
        self, timestamp_iso: Optional[str], sha: Optional[str]
    ) -> Optional[str]:
        if not timestamp_iso and not sha:
            return None
        return json.dumps({"since": timestamp_iso, "sha": sha})

    async def _events_from_commit(
        self, client: httpx.AsyncClient, commit: Dict[str, Any]
    ) -> List[IngestionEvent]:
        commit_sha = commit["sha"]
        commit_url = commit["url"]

        commit_response = await client.get(
            commit_url, headers=self.headers, timeout=30.0
        )
        commit_response.raise_for_status()
        commit_detail = commit_response.json()

        commit_message = commit_detail.get("commit", {}).get("message", "")
        commit_author = (
            commit_detail.get("commit", {}).get("author", {}).get("name", "unknown")
        )
        commit_date = (
            commit_detail.get("commit", {}).get("author", {}).get("date") or ""
        )
        commit_ts = (
            datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
            if commit_date
            else datetime.utcnow()
        )

        events: List[IngestionEvent] = []
        for file in commit_detail.get("files", []):
            filename = file.get("filename")
            if not filename or not self._is_docs_file(filename):
                continue
            status = file.get("status")
            event_type = self._map_status_to_event(status or "")
            if not event_type:
                continue

            events.append(
                self._create_event(
                    commit_sha=commit_sha,
                    filename=filename,
                    event_type=event_type,
                    commit_message=commit_message,
                    author=commit_author,
                    timestamp=commit_ts,
                )
            )
        return events
