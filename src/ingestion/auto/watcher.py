import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

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
    observer = Observer()
    handler = IngestHandler(base_dir)
    observer.schedule(handler, base_dir, recursive=True)
    observer.start()
    return observer
