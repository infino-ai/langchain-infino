"""LangChain integration for Infino."""

from importlib.metadata import PackageNotFoundError, version

# Re-exported so callers can handle engine failures without importing
# `infino` alongside this package. `ConflictError` and
# `ConnectionMemoryBudgetError` are both recoverable: back off and reissue,
# or narrow the request.
from infino import (
    ConflictError,
    ConnectionMemoryBudgetError,
    GcReport,
    InfinoError,
    MutationStats,
)

from langchain_infino.cache import InfinoSemanticCache
from langchain_infino.retrievers import InfinoBM25Retriever, InfinoHybridRetriever
from langchain_infino.translators import InfinoTranslator
from langchain_infino.vectorstores import InfinoVectorStore

try:
    __version__ = version("langchain-infino")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = [
    "InfinoVectorStore",
    "InfinoHybridRetriever",
    "InfinoBM25Retriever",
    "InfinoTranslator",
    "InfinoSemanticCache",
    "InfinoError",
    "ConflictError",
    "ConnectionMemoryBudgetError",
    "MutationStats",
    "GcReport",
]
