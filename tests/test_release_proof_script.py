import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_PROOF_SCRIPT = ROOT / "scripts" / "release_proof.sh"


def _script_text() -> str:
    return RELEASE_PROOF_SCRIPT.read_text()


def _release_proof_ingest_command() -> str:
    normalized = re.sub(r"\\\n\s*", " ", _script_text())
    for line in normalized.splitlines():
        if "scripts/ingestctl ingest" in line:
            return line.strip()
    raise AssertionError("release proof script must run scripts/ingestctl ingest")


def test_release_proof_uses_runnable_ingest_flow() -> None:
    command = _release_proof_ingest_command()
    assert "--watch" not in command, (
        "ingestctl --watch is currently a stub that exits 1; release proof "
        "must use an implemented ingest wait/polling path"
    )
    assert "--no-wait" not in command or "scripts/ingestctl status" in _script_text()
