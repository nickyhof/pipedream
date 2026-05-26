"""Column-templating for LLM ops.

LLM steps (e.g. ``classify``) carry a template string with ``{column}``
placeholders that are filled per row before the text is sent to the model. This
module is the single source of truth for which columns a template references
(so schema analysis can validate them) and how a template renders against a row.
"""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders(template: str) -> list[str]:
    """Column names referenced by a template, in first-seen order."""
    seen: list[str] = []
    for match in _PLACEHOLDER.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def render(template: str, row: dict[str, Any]) -> str:
    """Substitute ``{column}`` placeholders with that row's values."""
    return _PLACEHOLDER.sub(lambda m: str(row.get(m.group(1), "")), template)
