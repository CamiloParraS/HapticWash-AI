"""Adapter for OCDetect dataset."""

import pandas as pd

from .base import DatasetAdapter


class OCDetectAdapter(DatasetAdapter):
    """Ingest OCDetect haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize OCDetect data."""
        pass
