import os

import uvicorn
from fastapi import FastAPI

from .queue import enqueue_file
from .watcher import start_watcher

WATCH_DIR = os.getenv("INGEST_WATCH_DIR", "/app/ingest/incoming")
PORT = int(os.getenv("INGEST_PORT", "8081"))

app = FastAPI(title="Auto-Ingest Service")

observer = None


@app.on_event("startup")
async def startup():
    os.makedirs(WATCH_DIR, exist_ok=True)
    global observer
    observer = start_watcher(WATCH_DIR)


@app.on_event("shutdown")
async def shutdown():
    if observer:
        observer.stop()
        observer.join()


@app.get("/health")
async def health():
    return {"status": "ok", "watch_dir": WATCH_DIR}


@app.post("/enqueue")
async def enqueue(path: str):
    job_id = enqueue_file(path, source="api")
    return {"enqueued": job_id, "path": path}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
