"""Compatibility wrapper for routing logic."""
from typing import Dict, Optional

from app.rag.router import route_query as _route_query


def route_query(query: str) -> Dict[str, Optional[object]]:
    return _route_query(query)
