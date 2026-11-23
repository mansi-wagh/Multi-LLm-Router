import re

from app.config import CONFIG


_PRIVATE_PATTERNS = (
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?"
        r"[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)"
    ),
    re.compile(r"(?<!\d)(?:\d[\s-]?){11,15}\d(?!\d)"),
)
_GREETING_ONLY = re.compile(
    r"(?:hi+|hello+|hey+|howdy|good morning|good afternoon|good evening)"
    r"(?:\s+there)?[.!?]*",
    re.IGNORECASE,
)


def looks_private(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRIVATE_PATTERNS)


def matching_rule(text: str) -> tuple[str, str] | None:
    if looks_private(text):
        return CONFIG["private_label"], "private_pattern"
    if _GREETING_ONLY.fullmatch(text.strip()):
        return CONFIG["greeting_label"], "greeting"

    for label, keywords in CONFIG["keywords"].items():
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            if re.search(pattern, text, re.IGNORECASE):
                return label, f"keyword:{label}"
    return None


def route_by_rules(text: str) -> str | None:
    match = matching_rule(text)
    return match[0] if match else None
