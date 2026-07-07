"""docpipe -- Nutanix documentation PDF -> Markdown extraction pipeline.

Converts PDF product documentation into clean, ingestion-ready GitHub-Flavored
Markdown by rendering each page to an image and transcribing it with a
Qwen3.6-27B vision-language model served over an OpenAI-compatible API.

The pipeline is a standalone component: it PRODUCES a corpus that the separate
RAG ingestion path consumes. It never imports the RAG code and the RAG code
never changes to accept its output -- the only coupling is the on-disk Markdown
contract documented in ``docpipe/prompts.py`` and enforced in ``docpipe/clean.py``.
"""

__version__ = "0.1.0"

# The single source of truth for the pipeline_version stamped into every
# document's front matter. Bump when the extraction behaviour changes in a way
# that should invalidate cached page artifacts for provenance purposes.
PIPELINE_VERSION = __version__
