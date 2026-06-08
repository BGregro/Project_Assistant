"""
agent_tools/profile_updater.py  —  Phase 3g: User Profile Updater

Registers one tool:

  update_user_profile  — update or extend memory/user_profile.json
                         (top-level keys only; list fields support append via "+" prefix)

This tool IS destructive (modifies a user file) and requires explicit user
approval before execution, consistent with the permission layer convention.

Append semantics for list fields:
  field="skills", value="+TypeScript"  → appends "TypeScript" to skills list
  field="skills", value="Python"       → REPLACES skills with "Python" (string)
  field="goals",  value="+Earn money"  → appends "Earn money" to goals list

String fields are always replaced directly.

If memory/user_profile.json does not exist when this tool is called, a minimal
profile is created automatically so the update can succeed.
"""

import json
import logging
from pathlib import Path
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _profile_path() -> Path:
    """Absolute path to memory/user_profile.json (project root / memory /)."""
    return Path(__file__).parent.parent.parent / "memory" / "user_profile.json"


# ---------------------------------------------------------------------------
# Tool: update_user_profile
# ---------------------------------------------------------------------------

# Fields that are expected to be lists — these support the "+" append shorthand.
_LIST_FIELDS = frozenset([
    "skills", "accounts", "goals", "constraints",
    "current_projects", "interests", "languages",
])


async def update_user_profile(field: str, value: str) -> dict[str, Any]:
    """
    Update or add a top-level field in memory/user_profile.json.

    Args:
        field: Top-level key to update (e.g. "skills", "goals", "hardware").
        value: New value for the field.  For list fields, prefix with "+" to
               append an item rather than replacing the entire list.

    Returns:
        {"success": True, "field": ..., "new_value": ...}  on success
        {"success": False, "error": "..."}                 on failure
    """
    path = _profile_path()

    # ── Load existing profile (or start fresh) ────────────────────────────
    profile: dict[str, Any] = {}
    if path.exists():
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                logger.warning("[profile_updater] user_profile.json is not a dict — resetting.")
                profile = {}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"user_profile.json is invalid JSON: {e}"}
        except OSError as e:
            return {"success": False, "error": f"Could not read user_profile.json: {e}"}
    else:
        logger.info("[profile_updater] user_profile.json missing — will be created.")

    # ── Determine the update operation ───────────────────────────────────
    is_append = value.startswith("+") and field in _LIST_FIELDS
    updated_value: Any

    if is_append:
        # Append mode: strip the "+" prefix, then add to existing list (or start one)
        new_item = value[1:].strip()
        if not new_item:
            return {"success": False, "error": "Append value must not be empty after '+'."}

        existing = profile.get(field, [])
        if not isinstance(existing, list):
            # Field exists but is not a list — convert to list first
            existing = [str(existing)] if existing else []

        if new_item in existing:
            # Idempotent: item already present, no change needed
            return {
                "success":   True,
                "field":     field,
                "new_value": existing,
                "note":      f"'{new_item}' was already in {field!r} — no change made.",
            }

        updated_value = existing + [new_item]
        logger.info(f"[profile_updater] Appending '{new_item}' to field '{field}'.")
    else:
        # Replace mode — store the raw string (or try to parse JSON for complex values)
        # For simplicity we always store as a string; callers can use append for lists.
        updated_value = value
        logger.info(f"[profile_updater] Setting field '{field}' = {value!r}.")

    # ── Write updated profile ─────────────────────────────────────────────
    profile[field] = updated_value

    try:
        # Atomic write: write to .tmp first, then rename
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        logger.info(f"[profile_updater] user_profile.json updated (field: '{field}').")
    except OSError as e:
        return {"success": False, "error": f"Could not write user_profile.json: {e}"}

    return {
        "success":   True,
        "field":     field,
        "new_value": updated_value,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_profile_updater_tools() -> None:
    """Register update_user_profile into the global tool registry."""

    register_tool(
        name="update_user_profile",
        description=(
            "Update or add a field in the user's profile (memory/user_profile.json). "
            "Supports top-level keys only. "
            "For list fields (skills, goals, constraints, accounts, current_projects, "
            "interests, languages), prefix value with '+' to APPEND an item rather than "
            "replacing the entire list. "
            "Example: field='skills', value='+TypeScript' appends TypeScript to skills. "
            "Example: field='hardware', value='MacBook Pro M3' replaces the hardware string. "
            "Use this when the user tells you something new about themselves, their setup, "
            "or their goals, and asks you to remember it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "field": {
                    "type":        "string",
                    "description": (
                        "The top-level key to update in the user profile. "
                        "Common keys: name, skills, education, hardware, constraints, "
                        "goals, accounts, current_projects, interests, languages, available_time."
                    ),
                },
                "value": {
                    "type":        "string",
                    "description": (
                        "New value for the field. "
                        "For list fields, prefix with '+' to append a single item "
                        "(e.g. '+TypeScript'). Without '+', the field is replaced entirely."
                    ),
                },
            },
            "required": ["field", "value"],
        },
        handler=update_user_profile,
        is_destructive=True,   # Modifies a user file — requires approval
    )
