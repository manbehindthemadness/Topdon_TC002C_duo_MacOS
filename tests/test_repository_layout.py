"""
Keep application-owned Python modules small enough to review and maintain.
"""

from pathlib import Path


def test_application_python_files_stay_within_line_limit() -> None:
    """
    Enforce the project limit for source, examples, experiments, tests and tooling.
    """
    root = Path(__file__).resolve().parents[1]
    directories = ("src", "tests", "scripts", "tools", "examples", "experiments", ".agents")
    oversized = []
    for directory in directories:
        for path in (root / directory).rglob("*.py"):
            count = len(path.read_text(encoding="utf-8").splitlines())
            if count > 800:
                oversized.append(f"{path.relative_to(root)}: {count} lines")
    assert not oversized, "Split oversized modules by responsibility:\n" + "\n".join(oversized)
