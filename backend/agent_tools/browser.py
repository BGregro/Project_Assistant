"""
browser.py  —  Phase 3i: Read-Only Browser Automation

Registers three tools:

  browser_open(url)                 — navigate to a URL
  browser_read(selector, max_chars) — extract visible text from the current page
  browser_screenshot(filename)      — save a PNG of the current page

Uses Playwright async API with a module-level singleton browser context so the
browser persists (and stays warm) across multiple tool calls within a session.
The browser is closed automatically when the server shuts down.

If Playwright is not installed, registration is silently skipped — all other
tools continue to work normally.  The server never fails to start because of a
missing optional dependency.

Installation:
    pip install playwright
    playwright install chromium
"""

import json
import logging
import pathlib
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level browser singleton
# ---------------------------------------------------------------------------

_playwright_instance = None   # AsyncPlaywright context manager
_browser             = None   # Browser object
_page                = None   # Page object (shared across calls within a session)


async def _get_page():
    """
    Return the shared Page, lazily launching Playwright + Chromium on first call.

    Raises RuntimeError with a friendly install message if Playwright is not installed.
    Raises RuntimeError with the underlying error for any other launch failure.
    """
    global _playwright_instance, _browser, _page

    if _page is not None:
        # Already initialised — return the existing page
        return _page

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright not available. Run: pip install playwright && playwright install chromium"
        )

    try:
        _playwright_instance = async_playwright()
        pw = await _playwright_instance.__aenter__()
        _browser = await pw.chromium.launch(headless=True)
        _page    = await _browser.new_page()
        logger.info("[browser] Playwright/Chromium launched (headless).")
        return _page
    except Exception as exc:
        # Reset all globals so the next call can attempt initialisation again
        _playwright_instance = None
        _browser             = None
        _page                = None
        raise RuntimeError(f"[browser] Failed to launch browser: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_outputs_dir() -> pathlib.Path:
    """Read outputs_dir from config.json, default to 'outputs' if missing."""
    config_path = pathlib.Path(__file__).parent.parent.parent / "config.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        rel = cfg.get("outputs_dir", "outputs")
    except Exception:
        rel = "outputs"
    # Anchor to project root (two levels up from backend/agent_tools/)
    project_root = pathlib.Path(__file__).parent.parent.parent
    return project_root / rel


def _clean_text(raw: str) -> str:
    """Collapse runs of 3+ newlines into 2 and strip leading/trailing whitespace."""
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _browser_open(url: str) -> dict:
    """
    Navigate to *url* and wait for the DOM to be loaded.

    Args:
        url: The fully-qualified URL to navigate to, e.g. "https://example.com".

    Returns a dict:
        success  — True if navigation succeeded
        url      — the final URL after any redirects
        title    — page title
        error    — present only on failure
    """
    try:
        page = await _get_page()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        title = await page.title()
        final_url = page.url
        logger.info(f"[browser_open] Navigated to {final_url!r} (title: {title!r})")
        return {
            "success": True,
            "url":     final_url,
            "title":   title,
        }
    except Exception as exc:
        logger.warning(f"[browser_open] Failed to open {url!r}: {exc}")
        return {
            "success": False,
            "url":     url,
            "error":   str(exc),
        }


async def _browser_read(selector: str = "body", max_chars: int = 8000) -> dict:
    """
    Extract the visible text from the CSS *selector* on the current page.

    Falls back to 'body' if the specified selector is not found.
    Text longer than *max_chars* is truncated with a note.

    Args:
        selector:  CSS selector to read text from. Default: "body" (full page).
        max_chars: Maximum number of characters to return. Default: 8000.

    Returns a dict:
        success    — True on success
        selector   — the selector actually used (may differ from input if fallback used)
        content    — extracted text (cleaned and possibly truncated)
        char_count — character count AFTER truncation
        truncated  — True if the text was cut short
        error      — present only on failure
    """
    try:
        page = await _get_page()
    except RuntimeError as exc:
        return {"success": False, "selector": selector, "error": str(exc)}

    used_selector = selector
    try:
        raw_text = await page.inner_text(selector, timeout=5000)
    except Exception as selector_exc:
        # Selector not found or timeout — fall back to body
        logger.warning(
            f"[browser_read] Selector {selector!r} failed ({selector_exc}), "
            "falling back to 'body'."
        )
        used_selector = "body"
        try:
            raw_text = await page.inner_text("body", timeout=5000)
        except Exception as body_exc:
            return {
                "success":  False,
                "selector": used_selector,
                "error":    f"body fallback also failed: {body_exc}",
            }

    cleaned   = _clean_text(raw_text)
    truncated = len(cleaned) > max_chars
    content   = cleaned[:max_chars] + " [truncated]" if truncated else cleaned

    logger.info(
        f"[browser_read] selector={used_selector!r} "
        f"chars={len(content)} truncated={truncated}"
    )
    return {
        "success":    True,
        "selector":   used_selector,
        "content":    content,
        "char_count": len(content),
        "truncated":  truncated,
    }


async def _browser_screenshot(filename: str) -> dict:
    """
    Save a PNG screenshot of the current page to the outputs/ directory.

    Args:
        filename: Output filename, e.g. "screenshot.png".
                  Must match [a-zA-Z0-9_\\-.] and end in .png — anything else
                  is rejected to prevent path traversal attacks.

    Returns a dict:
        success — True on success
        path    — absolute path to the saved file
        error   — present only on failure
    """
    # ── Validate filename ──────────────────────────────────────────────
    if not re.match(r"^[a-zA-Z0-9_\-\.]+\.png$", filename):
        return {
            "success": False,
            "error": (
                f"Invalid filename {filename!r}. "
                "Use only letters, digits, underscores, hyphens, and dots. "
                "Must end in .png."
            ),
        }

    outputs_dir = _load_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_path = outputs_dir / filename

    try:
        page = await _get_page()
        await page.screenshot(path=str(save_path), full_page=False)
        logger.info(f"[browser_screenshot] Saved to {save_path}")
        return {
            "success": True,
            "path":    str(save_path),
        }
    except Exception as exc:
        logger.warning(f"[browser_screenshot] Failed: {exc}")
        return {
            "success": False,
            "error":   str(exc),
        }


# ---------------------------------------------------------------------------
# Shutdown helper (not a tool)
# ---------------------------------------------------------------------------

async def close_browser() -> None:
    """
    Close the Playwright browser and stop the Playwright instance.

    Called at server shutdown.  Swallows all exceptions — cleanup failures
    must never prevent a clean exit.
    """
    global _playwright_instance, _browser, _page

    if _page is not None:
        try:
            await _page.close()
            logger.info("[browser] Page closed.")
        except Exception as exc:
            logger.warning(f"[browser] Page close failed (non-fatal): {exc}")
        _page = None

    if _browser is not None:
        try:
            await _browser.close()
            logger.info("[browser] Browser closed.")
        except Exception as exc:
            logger.warning(f"[browser] Browser close failed (non-fatal): {exc}")
        _browser = None

    if _playwright_instance is not None:
        try:
            await _playwright_instance.__aexit__(None, None, None)
            logger.info("[browser] Playwright instance stopped.")
        except Exception as exc:
            logger.warning(f"[browser] Playwright stop failed (non-fatal): {exc}")
        _playwright_instance = None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Module-level flag so main.py can report browser availability in /status
browser_tools_registered = False


def register_browser_tools() -> None:
    """
    Register browser_open, browser_read, and browser_screenshot.

    Wrapped in a try/except ImportError so the server starts normally even
    if Playwright is not installed — in that case a warning is logged and
    registration is silently skipped.
    """
    global browser_tools_registered

    try:
        # Test the exact import _get_page() uses — the top-level 'playwright'
        # package may not be importable on all versions/platforms even when
        # the package is correctly installed.
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        logger.warning(
            "[browser] Playwright not installed — browser tools not registered. "
            "To enable: pip install playwright && playwright install chromium"
        )
        return

    from agent_tools import register_tool

    register_tool(
        name="browser_open",
        description=(
            "Navigate to a URL using a real Chromium browser (JavaScript rendered). "
            "Use this when fetch_page returns empty or incomplete content — many modern "
            "sites require JavaScript execution. Always call browser_open before "
            "browser_read or browser_screenshot."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The fully-qualified URL to navigate to, e.g. 'https://example.com'.",
                },
            },
            "required": ["url"],
        },
        handler=_browser_open,
        is_destructive=False,
    )

    register_tool(
        name="browser_read",
        description=(
            "Extract visible text from the current browser page. "
            "Use 'body' as the selector to read the full page, or a more specific "
            "CSS selector for targeted extraction (e.g. 'article', 'main', '#content'). "
            "Always call browser_open first to navigate to a page."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "CSS selector to read text from. Default: 'body' (full page). "
                        "Falls back to 'body' automatically if the selector is not found."
                    ),
                    "default": "body",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default: 8000.",
                    "default": 8000,
                },
            },
            "required": [],
        },
        handler=_browser_read,
        is_destructive=False,
    )

    register_tool(
        name="browser_screenshot",
        description=(
            "Save a PNG screenshot of the current browser page to the outputs/ directory. "
            "Useful for visually verifying what the browser loaded, or capturing a page "
            "for the user to review. Requires browser_open to have been called first."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Output filename, e.g. 'screenshot.png'. "
                        "Must use only letters, digits, underscores, hyphens, dots "
                        "and must end in .png."
                    ),
                },
            },
            "required": ["filename"],
        },
        handler=_browser_screenshot,
        is_destructive=True,   # Writes a file
    )

    browser_tools_registered = True
    logger.info("[browser] Registered: browser_open, browser_read, browser_screenshot")
