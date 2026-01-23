"""
app.rag package initializer with lightweight re-exports.

Provides stable imports for callers that previously used the flat layout,
while the actual implementation now lives in subpackages (pipeline/, router/, retriever/, etc.).
"""

from app.rag.pipeline.pipeline import RAGConfig, run_rag  # noqa: F401
from app.rag.router.router import route_query  # noqa: F401

__all__ = ["RAGConfig", "run_rag", "route_query"]
