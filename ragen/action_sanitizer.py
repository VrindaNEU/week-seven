import re

VALID_PATTERNS = [
    re.compile(r'^\s*search\[(.+?)\]\s*$', re.I),
    re.compile(r'^\s*click\[(\w[\w\-]+)\]\s*$', re.I),
    re.compile(r'^\s*buy(\s+now)?\s*$', re.I),
    re.compile(r'^\s*back\s*$', re.I),
]

BRACKET_PAT = re.compile(r'(?i)\b(search|click)\s*\[(.*?)\]')

def extract_first_valid_action(text: str) -> str | None:
    # Exact-line match
    for pat in VALID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0).strip().lower()

    # Embedded bracketed command
    m = BRACKET_PAT.search(text or "")
    if m:
        act, arg = m.group(1).lower(), m.group(2).strip()
        if act == "search" and arg:
            return f"search[{arg}]"
        if act == "click" and arg:
            return f"click[{arg}]"

    # Simple commands
    lt = (text or "").lower()
    if "buy now" in lt or re.search(r'(?i)^\s*buy\s*$', lt):
        return "buy now"
    if re.search(r'(?i)\bback\b', lt):
        return "back"
    return None

def sanitize(text: str, fallback_query: str) -> str:
    act = extract_first_valid_action(text or "")
    if act:
        return act
    # Safe fallback: always produce a legal action
    q = (fallback_query or "").strip()
    if not q:
        q = "best match"
    return f"search[{q}]"
