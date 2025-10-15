"""
GitHub connector for repository documentation ingestion.
Supports polling for changes and webhook-based real-time updates.
"""

import hashlib
import hmac
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
                # Fetch recent commits
                commits_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits"
                params = {"per_page": self.config.batch_size}
                if since:
                    params["since"] = since

                response = await client.get(
                    commits_url,
                    headers=self.headers,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                commits = response.json()

                if not commits:
                    logger.debug("No new commits found")
                    return [], since

                events = []
                latest_sha = commits[0]["sha"] if commits else since

                # Check each commit for docs changes
                for commit in commits:
                    commit_sha = commit["sha"]
                    commit_url = commit["url"]

                    # Fetch commit details to see changed files
                    commit_response = await client.get(
                        commit_url, headers=self.headers, timeout=30.0
                    )
                    commit_response.raise_for_status()
                    commit_detail = commit_response.json()

                    # Filter files in docs path
                    for file in commit_detail.get("files", []):
                        filename = file["filename"]

                        # Only process docs files
                        if not filename.startswith(self.docs_path):
                            continue
                        if not filename.endswith((".md", ".markdown", ".html")):
                            continue

                        # Determine event type
                        status = file["status"]
                        if status == "added":
                            event_type = "created"
                        elif status == "modified":
                            event_type = "updated"
                        elif status == "removed":
                            event_type = "deleted"
                        else:
                            continue

                        # Create ingestion event
                        event = IngestionEvent(
                            source_uri=f"https://github.com/{self.owner}/{self.repo}/blob/{commit_sha}/{filename}",
                            source_type="github",
                            event_type=event_type,
                            metadata={
                                "owner": self.owner,
                                "repo": self.repo,
                                "filename": filename,
                                "commit_sha": commit_sha,
                                "commit_message": commit.get("commit", {}).get(
                                    "message", ""
                                ),
                                "author": commit.get("commit", {})
                                .get("author", {})
                                .get("name", "unknown"),
                            },
                            timestamp=datetime.fromisoformat(
                                commit["commit"]["author"]["date"].replace(
                                    "Z", "+00:00"
                                )
                            ),
                        )
                        events.append(event)

                logger.info(
                    f"Fetched {len(events)} documentation changes from "
                    f"{len(commits)} commits"
                )
                return events, latest_sha

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"GitHub API error: {e.response.status_code} - "
                    f"{e.response.text}"
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

    async def webhook_to_event(
        self, payload: Dict[str, Any]
    ) -> Optional[IngestionEvent]:
        """
        Convert GitHub webhook payload to IngestionEvent.

        Processes 'push' events to detect docs changes.

        Args:
            payload: GitHub webhook payload

        Returns:
            IngestionEvent or None if no action needed
        """
        try:
            # Only process push events
            if "commits" not in payload:
                logger.debug("Ignoring non-push webhook event")
                return None

            # Extract commit info
            commits = payload.get("commits", [])
            if not commits:
                return None

            # Use most recent commit
            commit = commits[-1]
            commit_sha = commit["id"]

            # Check for docs changes
            added = commit.get("added", [])
            modified = commit.get("modified", [])
            removed = commit.get("removed", [])

            all_files = added + modified + removed

            # Filter for docs
            docs_files = [
                f
                for f in all_files
                if f.startswith(self.docs_path)
                and f.endswith((".md", ".markdown", ".html"))
            ]

            if not docs_files:
                logger.debug("No documentation files changed in push")
                return None

            # Create event for first changed doc
            filename = docs_files[0]
            if filename in removed:
                event_type = "deleted"
            elif filename in added:
                event_type = "created"
            else:
                event_type = "updated"

            event = IngestionEvent(
                source_uri=f"https://github.com/{self.owner}/{self.repo}/blob/{commit_sha}/{filename}",
                source_type="github",
                event_type=event_type,
                metadata={
                    "owner": self.owner,
                    "repo": self.repo,
                    "filename": filename,
                    "commit_sha": commit_sha,
                    "commit_message": commit.get("message", ""),
                    "author": commit.get("author", {}).get("username", "unknown"),
                    "webhook": True,
                },
                timestamp=datetime.fromisoformat(
                    commit["timestamp"].replace("Z", "+00:00")
                ),
            )

            logger.info(f"Webhook event created for {filename}: {event_type}")
            return event

        except Exception as e:
            logger.error(
                f"Error converting webhook payload to event: {e}",
                exc_info=True,
            )
            return None
