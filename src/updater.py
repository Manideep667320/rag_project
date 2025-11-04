"""File system watcher to trigger re-indexing when new files arrive.

Uses watchdog to monitor the `data/raw` directory and calls the provided
callback (by default `ingest.process_documents`) when files change.
"""
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable


class _ReindexHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable):
        super().__init__()
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory:
            print(f"Detected new file: {event.src_path}, triggering reindex")
            threading.Thread(target=self.callback, daemon=True).start()

    def on_modified(self, event):
        if not event.is_directory:
            print(f"Detected modified file: {event.src_path}, triggering reindex")
            threading.Thread(target=self.callback, daemon=True).start()


def start_watcher(path: str = "data/raw", callback: Callable = None):
    if callback is None:
        raise ValueError("Please provide a callback to run on changes (e.g. ingest.process_documents)")
    event_handler = _ReindexHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"Started watcher on {path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
