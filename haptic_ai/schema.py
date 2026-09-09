"""Canonical schema for haptic data."""
from pydantic import BaseModel, Field
import pandas as pd


class HapticRecord(BaseModel):
    """Single row in canonical haptic dataset."""
    pass


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize DataFrame to canonical schema."""
    return df
