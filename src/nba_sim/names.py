"""Player name matching across ESPN and the stats feeds."""

from __future__ import annotations

import unicodedata


def norm_person(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    for suffix in (" jr", " sr", " iii", " ii", " iv"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return "".join(char for char in text if char.isalpha())
