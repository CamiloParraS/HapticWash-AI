"""Adapter for Zhang WHO dataset."""

import pandas as pd

from .base import DatasetAdapter


class ZhangWhoAdapter(DatasetAdapter):
    """Ingest Zhang WHO haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize Zhang WHO data."""
        pass
