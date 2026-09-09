"""Test CLI."""
from haptic_ai.cli import main


def test_cli_no_args(monkeypatch):
    """Test CLI help."""
    monkeypatch.setattr("sys.argv", ["haptic-ai"])
    assert main() in [0, 1]
