"""Multi-agent research assistant built on LangGraph + Claude.

Public entry point: `run_query(question)` runs the full
router → retrieve → synthesize → critique → (retry) loop and returns
a final answer with citations and diagnostics.
"""

from .graph import run_query

__all__ = ["run_query"]
