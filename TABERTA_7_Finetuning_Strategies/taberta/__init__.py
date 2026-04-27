"""
TABERTA: A Language Model for Dataset Discovery
================================================

Structure-aware table embedding framework that adapts a bi-encoder
language model to relational data by fine-tuning on serialized tables.

Paper: TABERTA: A Language Model for Dataset Discovery (EDBT 2027)
"""

from taberta.data_loading import WikiDBsCorpus, DatabaseRecord, TableRecord
from taberta.serialization import schema_view, row_view, full_view
from taberta.config import (
    TrainingStrategy,
    PCConfig,
    SSCConfig,
    TCConfig,
    TCOptConfig,
    TCSBConfig,
    MLMConfig,
    HybridConfig,
)
from taberta.metrics import MetricsLogger

__all__ = [
    "WikiDBsCorpus",
    "DatabaseRecord",
    "TableRecord",
    "schema_view",
    "row_view",
    "full_view",
    "TrainingStrategy",
    "PCConfig",
    "SSCConfig",
    "TCConfig",
    "TCOptConfig",
    "TCSBConfig",
    "MLMConfig",
    "HybridConfig",
    "MetricsLogger",
]
