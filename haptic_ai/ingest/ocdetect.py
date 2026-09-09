"""Adapter for OCDetect dataset."""
from .base import DatasetAdapter
import pandas as pd


class OCDetectAdapter(DatasetAdapter):
    """Ingest OCDetect haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize OCDetect data."""
        pass
