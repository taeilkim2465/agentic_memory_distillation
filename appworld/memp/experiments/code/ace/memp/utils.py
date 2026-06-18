"""
Utility functions for the Mem^p module.
Self-contained — no dependency on the legacy ACE playbook/curator modules.
"""

import json
import re


def extract_json_from_text(text):
    """Extract the first valid JSON object from arbitrary LLM output."""
    if not text:
        return None

    # 1. Try the whole response as JSON (JSON-mode outputs)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json ... ``` blocks
    for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    # 3. Find JSON objects via balanced-brace scanning
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    if text[i] == "\\":
                        i += 1
                    i += 1
            i += 1
        if depth == 0:
            try:
                return json.loads(text[start:i])
            except json.JSONDecodeError:
                pass

    return None
