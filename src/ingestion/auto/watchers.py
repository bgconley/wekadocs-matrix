"""
Phase 6, Task 6.1: File System Watchers

Monitors file system directories for new documents using spool pattern.

Spool Pattern:
1. Write file as *.part (partial/in-progress)
2. Rename to *.ready when complete
3. Watcher only processes *.ready files
4. Prevents reading half-written files

See: /docs/pseudocode-phase6.md → 6.1
See: /docs/coder-guidance-phase6.md → 6.1
"""

import logging
import time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, Optional, Set

from .queue import JobQueue, compute_checksum

logger = logging.getLogger(__name__)


class FileSystemWatcher:
    """
    Watches file system directory for new documents

    Uses spool pattern: only processes files with .ready suffix
    """

    def __init__(
        self,
        watch_path: str,
        queue: JobQueue,
        tag: str = "default",
        debounce_seconds: float = 3.0,
        poll_interval: float = 5.0,
        recursive: bool = True,
    ):
        """
        Initialize FS watcher

        Args:
            watch_path: Directory to monitor
            queue: JobQueue instance for enqueueing jobs
            tag: Tag for documents from this watcher
            debounce_seconds: Wait time before processing (avoid rapid changes)
            poll_interval: How often to scan directory (seconds)
        """
        self.watch_path = Path(watch_path)
        self.queue = queue
        self.tag = tag
        self.debounce_seconds = debounce_seconds
        self.poll_interval = poll_interval
        self.recursive = recursive

        # Tracking
        self.seen_files: Set[str] = set()
        self.pending_files: Dict[str, float] = {}  # path -> first_seen_timestamp

        # Control
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        logger.info(f"FileSystemWatcher initialized: {watch_path} (tag={tag})")

    def start(self):
        """Start watcher in background thread"""
        if self._thread and self._thread.is_alive():
            logger.warning("Watcher already running")
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Watcher started: {self.watch_path}")

    def stop(self):
        """Stop watcher gracefully"""
        if not self._thread:
            return

        logger.info(f"Stopping watcher: {self.watch_path}")
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info(f"Watcher stopped: {self.watch_path}")

    def _watch_loop(self):
        """Main watch loop (runs in background thread)"""
        # Ensure watch directory exists
        self.watch_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Watching: {self.watch_path}")

        while not self._stop_event.is_set():
            try:
                self._scan_directory()
                self._process_pending()

            except Exception as e:
                logger.error(f"Error in watch loop: {e}", exc_info=True)

            # Sleep with interruptible waits
            for _ in range(int(self.poll_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _scan_directory(self):
        """Scan directory for new .ready files"""
        try:
            if self.recursive:
                ready_files = list(self.watch_path.rglob("*.ready"))
            else:
                ready_files = list(self.watch_path.glob("*.ready"))

            for ready_file in ready_files:
                try:
                    file_path = str(ready_file.resolve())
                except OSError:
                    # Fall back to absolute path if resolve fails
                    file_path = str(ready_file.absolute())

                # Skip if already processed
                if file_path in self.seen_files:
                    continue

                # Add to pending with timestamp (for debouncing)
                if file_path not in self.pending_files:
                    self.pending_files[file_path] = time.time()
                    logger.debug(f"Detected new file: {ready_file.name}")

        except Exception as e:
            logger.error(f"Error scanning directory: {e}")

    def _process_pending(self):
        """Process files that have passed debounce period"""
        now = time.time()
        to_process = []

        # Find files ready to process
        for file_path, first_seen in list(self.pending_files.items()):
            if now - first_seen >= self.debounce_seconds:
                to_process.append(file_path)
                del self.pending_files[file_path]

        # Process each file
        for file_path in to_process:
            try:
                self._process_file(file_path)
                self.seen_files.add(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

    def _process_file(self, file_path: str):
        """
        Process a single .ready file

        Args:
            file_path: Path to .ready file
        """
        path = Path(file_path)

        # Verify file still exists and is readable
        if not path.exists():
            logger.warning(f"File disappeared: {file_path}")
            return

        if not path.is_file():
            logger.warning(f"Not a file: {file_path}")
            return

        # Determine actual file path (strip .ready suffix)
        # Spool pattern: "file.md.ready" marker → actual file is "file.md"
        actual_path = path
        if path.name.endswith(".ready"):
            actual_path = path.with_name(path.name[:-6])  # Remove last 6 chars (.ready)

        # Verify actual file exists
        if not actual_path.exists():
            logger.warning(
                f"Actual file not found: {actual_path} (marker: {file_path})"
            )
            return

        # Compute checksum from the ACTUAL file, not the marker
        try:
            checksum = compute_checksum(str(actual_path))
        except Exception as e:
            logger.error(f"Failed to compute checksum for {actual_path}: {e}")
            return

        # Build URI pointing to the actual file
        source_uri = f"file://{actual_path.absolute()}"

        # Enqueue job
        job_id = self.queue.enqueue(
            source_uri=source_uri,
            checksum=checksum,
            tag=self.tag,
        )

        if job_id:
            logger.info(f"Enqueued {path.name} as job {job_id}")

            # Optionally archive the .ready file
            # (For now, just log; in production might move to processed/ folder)
            logger.debug(f"Processed .ready file: {file_path}")
        else:
            logger.info(f"Skipped duplicate file: {path.name}")


class S3Watcher:
    """
    Watches S3 bucket prefix for new objects

    NOTE: Requires boto3. Implement when S3 connector needed.
    """

    def __init__(self, bucket: str, prefix: str, queue: JobQueue, tag: str = "default"):
        self.bucket = bucket
        self.prefix = prefix
        self.queue = queue
        self.tag = tag
        logger.warning("S3Watcher not yet implemented")

    def start(self):
        raise NotImplementedError("S3Watcher requires boto3 and implementation")

    def stop(self):
        pass


class HTTPWatcher:
    """
    Polls HTTP endpoint for document list

    Endpoint should return JSON: {"documents": [{"url": "...", "checksum": "..."}]}
    """

    def __init__(
        self,
        endpoint_url: str,
        queue: JobQueue,
        tag: str = "default",
        poll_interval: float = 300.0,  # 5 minutes
    ):
        self.endpoint_url = endpoint_url
        self.queue = queue
        self.tag = tag
        self.poll_interval = poll_interval

        self.seen_checksums: Set[str] = set()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        logger.info(f"HTTPWatcher initialized: {endpoint_url}")

    def start(self):
        """Start HTTP polling in background thread"""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"HTTP watcher started: {self.endpoint_url}")

    def stop(self):
        """Stop HTTP watcher"""
        if not self._thread:
            return

        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info(f"HTTP watcher stopped: {self.endpoint_url}")

    def _poll_loop(self):
        """Poll HTTP endpoint periodically"""
        import requests

        while not self._stop_event.is_set():
            try:
                response = requests.get(self.endpoint_url, timeout=30)
                response.raise_for_status()

                data = response.json()
                documents = data.get("documents", [])

                for doc in documents:
                    url = doc.get("url")
                    checksum = doc.get("checksum")

                    if not url or not checksum:
                        continue

                    # Skip if already seen
                    if checksum in self.seen_checksums:
                        continue

                    # Enqueue
                    job_id = self.queue.enqueue(
                        source_uri=url,
                        checksum=checksum,
                        tag=self.tag,
                    )

                    if job_id:
                        self.seen_checksums.add(checksum)
                        logger.info(f"Enqueued HTTP document: {url}")

            except Exception as e:
                logger.error(f"HTTP poll error: {e}")

            # Sleep with interruptible waits
            for _ in range(int(self.poll_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)


class WatcherManager:
    """Manages multiple watchers"""

    def __init__(self, queue: JobQueue, config: Dict):
        """
        Initialize watcher manager

        Args:
            queue: JobQueue instance
            config: Configuration dict with watcher settings
        """
        self.queue = queue
        self.config = config
        self.watchers = []

        logger.info("WatcherManager initialized")

    def start_all(self):
        """Start all configured watchers"""
        # File system watchers
        if self.config.get("watch", {}).get("enabled", False):
            paths = self.config["watch"].get("paths", [])
            debounce = self.config["watch"].get("debounce_seconds", 3.0)
            poll = self.config["watch"].get("poll_interval", 5.0)
            recursive = self.config["watch"].get("recursive", True)

            for path in paths:
                watcher = FileSystemWatcher(
                    watch_path=path,
                    queue=self.queue,
                    tag=self.config.get("tag", "default"),
                    debounce_seconds=debounce,
                    poll_interval=poll,
                    recursive=recursive,
                )
                watcher.start()
                self.watchers.append(watcher)
                logger.info(f"Started FS watcher: {path}")

        # HTTP watchers
        if self.config.get("http", {}).get("enabled", False):
            endpoints = self.config["http"].get("endpoints", [])
            poll = self.config["http"].get("poll_interval", 300.0)

            for endpoint in endpoints:
                watcher = HTTPWatcher(
                    endpoint_url=endpoint,
                    queue=self.queue,
                    poll_interval=poll,
                )
                watcher.start()
                self.watchers.append(watcher)
                logger.info(f"Started HTTP watcher: {endpoint}")

        # S3 watchers (placeholder)
        if self.config.get("s3", {}).get("enabled", False):
            logger.warning("S3 watchers not yet implemented")

        logger.info(f"Started {len(self.watchers)} watchers")

    def stop_all(self):
        """Stop all watchers gracefully"""
        logger.info("Stopping all watchers...")
        for watcher in self.watchers:
            watcher.stop()
        self.watchers.clear()
        logger.info("All watchers stopped")
