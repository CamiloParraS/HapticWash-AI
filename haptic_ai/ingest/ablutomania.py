"""Adapter for Ablutomania dataset."""

import pandas as pd

from .base import DatasetAdapter


class AblutomaniaAdapter(DatasetAdapter):
    """Ingest Ablutomania haptic dataset."""

    def ingest(self) -> pd.DataFrame:
        """Load and normalize Ablutomania data."""
        pass
