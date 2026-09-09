"""Adapter for Ablutomania dataset."""
from .base import DatasetAdapter
import pandas as pd


class AblutomaniaAdapter(DatasetAdapter):
    """Ingest Ablutomania haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize Ablutomania data."""
        pass
