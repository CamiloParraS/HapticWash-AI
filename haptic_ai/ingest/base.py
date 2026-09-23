"""Base adapter for data ingestion."""

from abc import ABC, abstractmethod

import pandas as pd


class DatasetAdapter(ABC):
    """ABC for dataset adapters."""

    @abstractmethod
    def ingest(self) -> pd.DataFrame:
        """Yield canonical DataFrames."""
        pass
