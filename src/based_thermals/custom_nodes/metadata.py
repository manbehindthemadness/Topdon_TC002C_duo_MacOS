"""
Read embedded custom-node metadata without importing Python packages.
"""

import ast
import json
from functools import lru_cache
from typing import Any

from .configuration import configuration
from .models import model_source


@lru_cache(maxsize=16)
def declarations(source: str) -> tuple[tuple[str, str], ...]:
    """
    Extract literal CONFIG_JSON and MODEL_SOURCE assignments at module scope.
    """
    found: dict[str, str] = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            targets, expression = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, expression = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in ("CONFIG_JSON", "MODEL_SOURCE"):
                continue
            if target.id in found:
                raise ValueError(f"Duplicate custom declaration: {target.id}")
            try:
                if expression is None:
                    raise ValueError("Declaration needs a literal value")
                value: Any = ast.literal_eval(expression)
                if target.id == "CONFIG_JSON":
                    text = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
                    configuration(text)
                else:
                    model_source(value)
                    text = json.dumps(value, allow_nan=False)
            except (ValueError, TypeError, RecursionError) as exc:
                raise ValueError(f"{target.id} must be valid literal metadata: {exc}") from exc
            found[target.id] = text
    return tuple(found.items())


def embedded_metadata(source: str) -> dict[str, str]:
    """
    Copy validated metadata without exposing mutable cached declarations.
    """
    return dict(declarations(source))
