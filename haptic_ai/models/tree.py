"""Feature-based baselines (SPEC M2): Random Forest, Gradient Boosting, and the dummy."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def make(name: str, seed: int):
    """A fresh, seeded, unfitted model. ``dummy`` is the chance level (SPEC 9.4)."""
    if name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gb":
        # ponytail: no class_weight; it disables histogram subtraction (4x slower) and the
        # step classes are within 2x of each other. Revisit if GB wins and a class lags.
        return HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
    raise ValueError(f"unknown model {name!r}")
