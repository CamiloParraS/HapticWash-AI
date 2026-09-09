"""Adapter for Zhang WHO dataset."""
from .base import DatasetAdapter
import pandas as pd


class ZhangWhoAdapter(DatasetAdapter):
    """Ingest Zhang WHO haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize Zhang WHO data."""
        pass
