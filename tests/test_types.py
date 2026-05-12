"""Type-checking test — verifies mypy passes on core modules."""

import subprocess
import sys


def test_mypy_passes():
    """Run mypy on core agent modules (non-strict to allow gradual adoption)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy",
            "agent/state.py",
            "agent/router.py",
            "agent/query_rewriter.py",
            "agent/answer.py",
            "agent/clarify.py",
            "agent/refuse.py",
            "agent/tools.py",
            "observability/tracer.py",
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy found errors:\n{result.stdout}\n{result.stderr}"
    )
