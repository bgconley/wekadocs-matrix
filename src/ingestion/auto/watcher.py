"""
Deprecated minimal watchdog-based watcher.

This module was an early Phase 6 MVP. The production watcher is
`FileSystemWatcher` in `src.ingestion.auto.watchers`, which supports:
- spool `.ready` mode (production default)
- direct mode (dev convenience)
- polling resilience + checksum dedup + debounce

`ingestion-service` no longer imports this module.
"""

import os
import warnings

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from .queue import enqueue_file

ALLOWED_EXT = {".md", ".markdown", ".html", ".htm"}


class IngestHandler(FileSystemEventHandler):
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        ext = os.path.splitext(path)[1].lower()
        if ext in ALLOWED_EXT:
            enqueue_file(path, source="watcher")


def start_watcher(base_dir: str):
    warnings.warn(
        "start_watcher() is deprecated; use FileSystemWatcher in watchers.py",
        DeprecationWarning,
        stacklevel=2,
    )
    observer = Observer()
    handler = IngestHandler(base_dir)
    observer.schedule(handler, base_dir, recursive=True)
    observer.start()
    return observer
