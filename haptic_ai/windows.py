"""Sliding windows + majority-vote labels + subject grouping."""
import pandas as pd


def create_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Create sliding windows from time series."""
    return df


def majority_vote_labels(windows, labels):
    """Aggregate labels via majority voting."""
    pass


def group_by_subject(df: pd.DataFrame):
    """Group windows by subject."""
    pass
