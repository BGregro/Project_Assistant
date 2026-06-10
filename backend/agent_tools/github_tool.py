"""
github_tool.py  —  Phase 5a: GitHub Integration

Provides six tools for managing GitHub repositories via the GitHub REST API:

  github_list_repos    — list the authenticated user's repositories
  github_create_repo   — create a new repository (destructive)
  github_push_file     — push a single file to a repository (destructive)
  github_read_file     — read a file from a repository
  github_list_files    — list files/directories in a repository path
  github_create_issue  — create an issue in a repository (destructive)

All tools read the GitHub personal access token from the GITHUB_TOKEN
environment variable.  If the token is not set, every tool returns a
descriptive error with instructions to create one — no crashes.

Setup:
  1. Create a token at https://github.com/settings/tokens with 'repo' scope.
  2. Add GITHUB_TOKEN=your_token_here to your .env file.

Security note: github_create_repo, github_push_file, and github_create_issue
are marked is_destructive=True — they require user confirmation before running.
"""

from __future__ import annotations

import base64
import logging
import os

import httpx

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"

_NO_TOKEN_ERROR = {
    "success": False,
    "error": (
        "GITHUB_TOKEN not set in .env file. "
        "Create a token at https://github.com/settings/tokens with 'repo' scope, "
        "then add GITHUB_TOKEN=your_token_here to your .env file and restart the server."
    ),
}


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

async def _gh(method: str, path: str, **kwargs) -> tuple[bool, dict]:
    """
    Make an authenticated GitHub API request.

    Args:
        method: HTTP method (GET, POST, PUT, etc.)
        path:   API path, e.g. '/user/repos' — will be appended to GITHUB_API_BASE.
        **kwargs: Additional arguments passed to httpx.AsyncClient.request().

    Returns:
        (True, response_json)   on 2xx status
        (False, {"error": status_code, "message": "..."}) on any non-2xx status or exception
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return False, {"error": "no_token", "message": "GITHUB_TOKEN not set"}

    url = GITHUB_API_BASE + path
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            logger.debug(f"[github] {method} {path} → {response.status_code}")

            # Parse JSON response (may be empty on some 2xx responses)
            try:
                body = response.json()
            except Exception:
                body = {}

            if response.is_success:
                return True, body
            else:
                message = body.get("message", "") if isinstance(body, dict) else str(body)
                return False, {"error": response.status_code, "message": message}

    except httpx.TimeoutException:
        logger.warning(f"[github] Request timed out: {method} {path}")
        return False, {"error": "timeout", "message": f"Request to {path} timed out after 30s"}
    except Exception as e:
        logger.error(f"[github] Unexpected error for {method} {path}: {e}")
        return False, {"error": "exception", "message": str(e)}


# ---------------------------------------------------------------------------
# Tool: github_list_repos
# ---------------------------------------------------------------------------

async def github_list_repos(visibility: str = "all") -> dict:
    """
    List the authenticated user's GitHub repositories.

    Args:
        visibility: One of "all", "public", or "private". Defaults to "all".

    Returns:
        {
            "success": True,
            "repos": [{"name", "full_name", "description", "private", "url",
                       "updated_at", "language"}, ...]
        }
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    # Normalise visibility
    if visibility not in {"all", "public", "private"}:
        visibility = "all"

    ok, data = await _gh("GET", f"/user/repos?visibility={visibility}&sort=updated&per_page=20")
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    repos = [
        {
            "name":        r.get("name", ""),
            "full_name":   r.get("full_name", ""),
            "description": r.get("description") or "",
            "private":     r.get("private", False),
            "url":         r.get("html_url", ""),
            "updated_at":  r.get("updated_at", ""),
            "language":    r.get("language") or "",
        }
        for r in (data if isinstance(data, list) else [])
    ]

    return {"success": True, "repos": repos, "count": len(repos)}


# ---------------------------------------------------------------------------
# Tool: github_create_repo
# ---------------------------------------------------------------------------

async def github_create_repo(
    name: str,
    description: str = "",
    private: bool = True,
) -> dict:
    """
    Create a new GitHub repository for the authenticated user.

    The repository is initialised with an automatic first commit so it's
    immediately pushable without additional setup.

    Args:
        name:        Repository name (e.g. "my-project").
        description: Optional short description shown on GitHub.
        private:     Whether the repository should be private. Defaults to True.

    Returns:
        {"success": True, "name", "full_name", "url", "clone_url"}
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    payload = {
        "name":        name.strip(),
        "description": description.strip(),
        "private":     private,
        "auto_init":   True,  # creates initial commit so repo is immediately pushable
    }

    ok, data = await _gh("POST", "/user/repos", json=payload)
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    return {
        "success":   True,
        "name":      data.get("name", name),
        "full_name": data.get("full_name", ""),
        "url":       data.get("html_url", ""),
        "clone_url": data.get("clone_url", ""),
    }


# ---------------------------------------------------------------------------
# Tool: github_push_file
# ---------------------------------------------------------------------------

async def github_push_file(
    repo: str,
    path: str,
    content: str,
    message: str = "",
    branch: str = "main",
) -> dict:
    """
    Push a single file to a GitHub repository.

    Automatically detects whether the file already exists (update) or is new
    (create) by fetching its current SHA before writing.

    Args:
        repo:    Repository full name, e.g. "username/my-project".
        path:    File path within the repo, e.g. "src/main.py".
        content: Full file content as a string (UTF-8).
        message: Commit message. Auto-generated if omitted.
        branch:  Target branch. Defaults to "main".

    Returns:
        {"success": True, "path", "repo", "url", "action": "created"|"updated"}
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    # ── Step 1: Check if the file already exists to get its SHA ──────────────
    existing_sha: str | None = None
    ok, existing = await _gh("GET", f"/repos/{repo}/contents/{path}?ref={branch}")
    if ok and isinstance(existing, dict):
        existing_sha = existing.get("sha")

    # ── Step 2: Build the PUT payload ─────────────────────────────────────────
    # GitHub requires base64-encoded content
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

    # Auto-generate commit message if not provided
    if not message:
        action_word = "Update" if existing_sha else "Add"
        message = f"{action_word} {path}"

    payload: dict = {
        "message": message,
        "content": encoded,
        "branch":  branch,
    }
    # Include SHA only when updating — omitting it means "create new file"
    if existing_sha:
        payload["sha"] = existing_sha

    # ── Step 3: PUT the file ──────────────────────────────────────────────────
    ok, data = await _gh("PUT", f"/repos/{repo}/contents/{path}", json=payload)
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    action   = "updated" if existing_sha else "created"
    file_url = data.get("content", {}).get("html_url", "")

    logger.info(f"[github] File {action}: {repo}/{path} (branch: {branch})")
    return {
        "success": True,
        "path":    path,
        "repo":    repo,
        "url":     file_url,
        "action":  action,
    }


# ---------------------------------------------------------------------------
# Tool: github_read_file
# ---------------------------------------------------------------------------

async def github_read_file(
    repo: str,
    path: str,
    branch: str = "main",
) -> dict:
    """
    Read a file from a GitHub repository.

    Files larger than 100KB are truncated to their first 50KB with a note.

    Args:
        repo:   Repository full name, e.g. "username/my-project".
        path:   File path within the repo, e.g. "src/main.py".
        branch: Branch to read from. Defaults to "main".

    Returns:
        {"success": True, "path", "content": str, "size", "sha", "url"}
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    ok, data = await _gh("GET", f"/repos/{repo}/contents/{path}?ref={branch}")
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    if not isinstance(data, dict):
        return {"success": False, "error": "Unexpected response format from GitHub API"}

    # Files are returned as base64-encoded content
    raw_b64   = data.get("content", "").replace("\n", "")
    file_size = data.get("size", 0)

    truncated = False
    try:
        decoded = base64.b64decode(raw_b64).decode("utf-8", errors="replace")
        # Truncate very large files to avoid flooding context
        if len(decoded) > 100_000:
            decoded   = decoded[:50_000]
            truncated = True
    except Exception as e:
        return {"success": False, "error": f"Could not decode file content: {e}"}

    result: dict = {
        "success": True,
        "path":    path,
        "content": decoded,
        "size":    file_size,
        "sha":     data.get("sha", ""),
        "url":     data.get("html_url", ""),
    }
    if truncated:
        result["truncated"] = True
        result["note"]      = f"File is {file_size} bytes — only first 50KB returned."

    return result


# ---------------------------------------------------------------------------
# Tool: github_list_files
# ---------------------------------------------------------------------------

async def github_list_files(
    repo: str,
    path: str = "",
    branch: str = "main",
) -> dict:
    """
    List files and directories at a path in a GitHub repository.

    Args:
        repo:   Repository full name, e.g. "username/my-project".
        path:   Directory path within the repo. Defaults to root ("").
        branch: Branch to inspect. Defaults to "main".

    Returns:
        {"success": True, "path", "items": [{"name", "path", "type", "size"}, ...]}
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    api_path = f"/repos/{repo}/contents/{path}?ref={branch}" if path else f"/repos/{repo}/contents?ref={branch}"
    ok, data = await _gh("GET", api_path)
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    if not isinstance(data, list):
        # Single file returned — wrap it
        if isinstance(data, dict):
            data = [data]
        else:
            return {"success": False, "error": "Unexpected response format"}

    items = [
        {
            "name": item.get("name", ""),
            "path": item.get("path", ""),
            "type": item.get("type", ""),   # "file" or "dir"
            "size": item.get("size", 0),
        }
        for item in data
    ]

    return {"success": True, "path": path or "/", "items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Tool: github_create_issue
# ---------------------------------------------------------------------------

async def github_create_issue(
    repo: str,
    title: str,
    body: str = "",
    labels: str = "",
) -> dict:
    """
    Create an issue in a GitHub repository.

    Args:
        repo:   Repository full name, e.g. "username/my-project".
        title:  Issue title.
        body:   Issue description (markdown supported).
        labels: Comma-separated label names to apply, e.g. "bug,help wanted".
                Labels that don't exist in the repo will be ignored by GitHub.

    Returns:
        {"success": True, "number", "title", "url"}
    """
    if not os.getenv("GITHUB_TOKEN"):
        return _NO_TOKEN_ERROR

    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []

    payload: dict = {
        "title": title.strip(),
        "body":  body.strip(),
    }
    if label_list:
        payload["labels"] = label_list

    ok, data = await _gh("POST", f"/repos/{repo}/issues", json=payload)
    if not ok:
        return {"success": False, "error": data.get("message", str(data))}

    return {
        "success": True,
        "number":  data.get("number", 0),
        "title":   data.get("title", title),
        "url":     data.get("html_url", ""),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_github_tools() -> None:
    """Register all six GitHub tools with the live tool registry."""

    register_tool(
        name="github_list_repos",
        description=(
            "List the authenticated user's GitHub repositories, sorted by most recently updated. "
            "Use to discover existing repos before creating new ones or to check if a project "
            "has already been pushed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "visibility": {
                    "type":        "string",
                    "description": "Filter by visibility: 'all' (default), 'public', or 'private'.",
                    "enum":        ["all", "public", "private"],
                },
            },
        },
        handler=github_list_repos,
        is_destructive=False,
    )

    register_tool(
        name="github_create_repo",
        description=(
            "Create a new GitHub repository for the authenticated user. "
            "The repo is auto-initialised with a first commit so it's immediately pushable. "
            "Use after completing and testing a project to set up its remote home."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type":        "string",
                    "description": "Repository name (e.g. 'my-project'). Use hyphens, not underscores.",
                },
                "description": {
                    "type":        "string",
                    "description": "Short description shown on GitHub (one sentence).",
                },
                "private": {
                    "type":        "boolean",
                    "description": "Whether the repo should be private. Defaults to true.",
                },
            },
            "required": ["name"],
        },
        handler=github_create_repo,
        is_destructive=True,
    )

    register_tool(
        name="github_push_file",
        description=(
            "Push a single file to a GitHub repository. "
            "Automatically creates or updates the file as needed. "
            "Push files in implementation_order from the scaffold. "
            "Call once per file when pushing a completed project."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {
                    "type":        "string",
                    "description": "Repository full name, e.g. 'username/my-project'.",
                },
                "path": {
                    "type":        "string",
                    "description": "File path within the repo, e.g. 'src/main.py'.",
                },
                "content": {
                    "type":        "string",
                    "description": "Full file content as a string.",
                },
                "message": {
                    "type":        "string",
                    "description": "Commit message. Auto-generated if omitted.",
                },
                "branch": {
                    "type":        "string",
                    "description": "Target branch. Defaults to 'main'.",
                },
            },
            "required": ["repo", "path", "content"],
        },
        handler=github_push_file,
        is_destructive=True,
    )

    register_tool(
        name="github_read_file",
        description=(
            "Read a file from a GitHub repository. "
            "Returns decoded content as a string. "
            "Files over 100KB are truncated to their first 50KB."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {
                    "type":        "string",
                    "description": "Repository full name, e.g. 'username/my-project'.",
                },
                "path": {
                    "type":        "string",
                    "description": "File path within the repo, e.g. 'README.md'.",
                },
                "branch": {
                    "type":        "string",
                    "description": "Branch to read from. Defaults to 'main'.",
                },
            },
            "required": ["repo", "path"],
        },
        handler=github_read_file,
        is_destructive=False,
    )

    register_tool(
        name="github_list_files",
        description=(
            "List files and directories at a path in a GitHub repository. "
            "Defaults to the repository root. "
            "Use to explore an existing repo's structure before reading files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {
                    "type":        "string",
                    "description": "Repository full name, e.g. 'username/my-project'.",
                },
                "path": {
                    "type":        "string",
                    "description": "Directory path within the repo. Defaults to root.",
                },
                "branch": {
                    "type":        "string",
                    "description": "Branch to inspect. Defaults to 'main'.",
                },
            },
            "required": ["repo"],
        },
        handler=github_list_files,
        is_destructive=False,
    )

    register_tool(
        name="github_create_issue",
        description=(
            "Create an issue in a GitHub repository. "
            "Useful for tracking bugs, feature requests, or TODOs discovered during development. "
            "Labels must already exist in the repository to be applied."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {
                    "type":        "string",
                    "description": "Repository full name, e.g. 'username/my-project'.",
                },
                "title": {
                    "type":        "string",
                    "description": "Issue title (concise, one line).",
                },
                "body": {
                    "type":        "string",
                    "description": "Issue description. Markdown is supported.",
                },
                "labels": {
                    "type":        "string",
                    "description": "Comma-separated label names, e.g. 'bug,help wanted'.",
                },
            },
            "required": ["repo", "title"],
        },
        handler=github_create_issue,
        is_destructive=True,
    )

    logger.info(
        "[startup] Registered GitHub tools: github_list_repos, github_create_repo, "
        "github_push_file, github_read_file, github_list_files, github_create_issue"
    )
