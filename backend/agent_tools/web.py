"""
agent_tools/web.py  —  Web Tools (Phase 2)

Provides two tools the agent can use:
  - search_web:   Search the web via SearXNG (self-hosted) with an automatic
                  DuckDuckGo HTML fallback — no API key required.
  - fetch_page:   Fetch the plain-text content of any URL via httpx.

Both tools are non-destructive (no side effects on the user's system).

search_web strategy:
  1. Try SearXNG at the URL configured in config.json → tools.web.searxng_url
     (default http://localhost:8888).  SearXNG must be running locally — see
     SEARXNG_SETUP.md in the project root for Docker setup instructions.
  2. If SearXNG is unreachable or returns a non-200 status, silently fall back
     to scraping DuckDuckGo's HTML interface using only the stdlib html.parser.
  3. If both fail, return a clear error dict.

fetch_page uses a 10-second timeout and gracefully handles connection
errors so a slow/unreachable server cannot block the agent loop.
"""

import html
import html.parser
import logging
import re
from typing import Any

import httpx

from . import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config access — read tool settings from config.json at call time
# so live config changes (e.g. via set_config) are always respected.
# ---------------------------------------------------------------------------

def _get_web_config() -> dict:
    """
    Load web-tool settings from config.json → tools.web.
    Returns sensible defaults if the key is missing or the file is unreadable.
    """
    import json
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config.json"
    defaults = {
        "max_results": 5,
        "fetch_timeout": 10.0,
        "max_fetch_chars": 8000,
        "searxng_url": "http://localhost:8888",
    }
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return {**defaults, **cfg.get("tools", {}).get("web", {})}
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# HTML stripping helper
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    """
    Remove HTML tags and decode HTML entities from a string.

    Uses only the Python standard library (html module + re) so we avoid
    adding beautifulsoup4 as a dependency at this phase.

    Strategy:
      1. Remove <script> and <style> blocks (and their content) entirely —
         these are never useful as plain text.
      2. Replace block-level tags with newlines to preserve visual structure.
      3. Strip all remaining tags.
      4. Decode HTML entities (e.g. &amp; → &, &nbsp; → space).
      5. Collapse excessive blank lines and leading/trailing whitespace.
    """
    # Remove <script>…</script> and <style>…</style> blocks
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)

    # Replace block-level tags with newlines so paragraphs don't merge
    block_tags = r"</?(?:p|div|br|h[1-6]|li|tr|blockquote|pre|section|article)[^>]*>"
    raw = re.sub(block_tags, "\n", raw, flags=re.IGNORECASE)

    # Strip all remaining tags
    raw = re.sub(r"<[^>]+>", "", raw)

    # Decode HTML entities (&amp; &lt; &nbsp; etc.)
    raw = html.unescape(raw)

    # Collapse runs of blank lines (> 2) and strip leading/trailing whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# DDG HTML parser  (stdlib only — no beautifulsoup4)
# ---------------------------------------------------------------------------

class _DDGParser(html.parser.HTMLParser):
    """
    Minimal SAX-style parser for DuckDuckGo's HTML result page.

    DDG's structure (simplified):
        <div class="result">
            ...
            <a class="result__a" href="...">Title text</a>
            ...
            <a class="result__snippet">Snippet text</a>
            ...
        </div>

    We walk the tag stream and collect (title, url, snippet) triples.
    The parser is intentionally simple — it only needs to survive DDG's
    actual output, not arbitrary HTML.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []

        # State machine flags
        self._in_result_a    = False   # inside <a class="result__a">
        self._in_snippet_a   = False   # inside <a class="result__snippet">
        self._current_title  = ""
        self._current_url    = ""
        self._current_snippet = ""

    # ------------------------------------------------------------------
    # html.parser callbacks
    # ------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr = dict(attrs)
        css  = attr.get("class", "")

        if tag == "div" and "result" in css.split():
            # Start of a new result block — flush previous if populated
            self._flush()

        elif tag == "a":
            if "result__a" in css.split():
                self._in_result_a  = True
                self._current_url  = attr.get("href", "")
                self._current_title = ""
            elif "result__snippet" in css.split():
                self._in_snippet_a   = True
                self._current_snippet = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_result_a  = False
            self._in_snippet_a = False

    def handle_data(self, data: str) -> None:
        if self._in_result_a:
            self._current_title += data
        elif self._in_snippet_a:
            self._current_snippet += data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Save the current result if it has at least a URL and title."""
        if self._current_url and self._current_title.strip():
            self.results.append({
                "title":       self._current_title.strip(),
                "url":         self._current_url,
                "description": self._current_snippet.strip(),
            })
        self._current_title   = ""
        self._current_url     = ""
        self._current_snippet = ""
        self._in_result_a     = False
        self._in_snippet_a    = False

    def close(self) -> None:
        """Flush the last result before closing."""
        self._flush()
        super().close()


# ---------------------------------------------------------------------------
# SearXNG helper
# ---------------------------------------------------------------------------

async def _search_searxng(
    query: str,
    count: int,
    base_url: str,
) -> list[dict] | None:
    """
    Query a local SearXNG instance and return parsed results.

    Returns a list of result dicts on success, or None if SearXNG is
    unreachable / returns a non-200 status (caller will fall back to DDG).

    Each result dict has: title, url, description.
    """
    params = {
        "q":          query,
        "format":     "json",
        "categories": "general",
        "language":   "en",
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"[web] SearXNG unavailable ({base_url}): {e}")
        return None
    except Exception as e:
        logger.warning(f"[web] SearXNG unexpected error: {e}")
        return None

    raw = data.get("results", [])
    results = [
        {
            "title":       r.get("title", ""),
            "url":         r.get("url", ""),
            "description": r.get("content", ""),   # SearXNG uses "content" for snippets
        }
        for r in raw[:count]
    ]
    logger.info(f"[web] SearXNG returned {len(results)} results for {query!r}")
    return results


# ---------------------------------------------------------------------------
# DuckDuckGo fallback helper
# ---------------------------------------------------------------------------

async def _search_ddg(query: str, count: int) -> list[dict] | None:
    """
    Scrape DuckDuckGo's HTML interface and return parsed results.

    Uses stdlib html.parser via _DDGParser — no beautifulsoup4.
    Returns a list of result dicts on success, or None on failure.

    Each result dict has: title, url, description.
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {
        # A browser-like UA is needed; DDG blocks obvious bot strings
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    params = {"q": query}

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers=headers,
        ) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw_html = resp.text
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(f"[web] DDG fallback failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"[web] DDG unexpected error: {e}")
        return None

    parser = _DDGParser()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception as e:
        # html.parser.HTMLParser.feed() can raise on truly broken HTML
        logger.warning(f"[web] DDG HTML parse error: {e}")
        return None

    results = parser.results[:count]
    logger.info(f"[web] DDG fallback returned {len(results)} results for {query!r}")
    return results


# ---------------------------------------------------------------------------
# Tool: search_web
# ---------------------------------------------------------------------------

async def search_web(query: str, n_results: int = 5) -> dict[str, Any]:
    """
    Search the web — tries SearXNG first, falls back to DuckDuckGo.

    SearXNG (primary):
        Sends a JSON search request to the locally-running SearXNG instance
        configured at config.json → tools.web.searxng_url
        (default http://localhost:8888).
        See SEARXNG_SETUP.md for Docker setup instructions.

    DuckDuckGo (fallback):
        If SearXNG is unreachable or returns an error, the tool automatically
        scrapes DuckDuckGo's HTML interface using stdlib html.parser.
        No API key or external service account needed.

    Args:
        query:     The search query string.
        n_results: Maximum number of results to return (default 5, max 20).

    Returns a list of results, each containing:
        title       — page title
        url         — canonical URL
        description — short snippet
    """
    cfg   = _get_web_config()
    count = min(n_results, int(cfg.get("max_results", 5)), 20)
    searxng_url = cfg.get("searxng_url", "http://localhost:8888").rstrip("/")

    logger.info(f"[web] search_web: query={query!r}, count={count}")

    # --- 1. Try SearXNG ---
    results = await _search_searxng(query, count, searxng_url)

    # --- 2. DDG fallback ---
    if results is None:
        logger.info("[web] Falling back to DuckDuckGo HTML scrape.")
        results = await _search_ddg(query, count)

    # --- 3. Both failed ---
    if results is None:
        return {
            "success": False,
            "error": (
                "Both SearXNG and DuckDuckGo are unavailable. "
                "Check that SearXNG is running (see SEARXNG_SETUP.md) "
                "or that you have internet access for the DDG fallback."
            ),
        }

    return {"success": True, "query": query, "results": results}


# ---------------------------------------------------------------------------
# Tool: fetch_page
# ---------------------------------------------------------------------------

async def fetch_page(url: str) -> dict[str, Any]:
    """
    Fetch the plain-text content of a URL.

    - Uses httpx with a 10-second timeout.
    - Strips HTML tags and decodes entities using stdlib only.
    - Truncates output to max_fetch_chars (default 8000) with a note if cut.

    Args:
        url: The full URL to fetch (must start with http:// or https://).

    Returns:
        success  — bool
        url      — the URL that was fetched
        content  — plain-text content (possibly truncated)
    """
    cfg = _get_web_config()
    timeout   = float(cfg["fetch_timeout"])
    max_chars = int(cfg["max_fetch_chars"])

    # Basic sanity check — only allow http/https to prevent SSRF via file:// etc.
    if not url.startswith(("http://", "https://")):
        return {
            "success": False,
            "url": url,
            "content": "Invalid URL: only http:// and https:// are supported.",
        }

    logger.info(f"[web] fetch_page: {url}")

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            # Some servers reject bots; a neutral UA avoids most blocks
            headers={"User-Agent": "Mozilla/5.0 (compatible; PersonalAgent/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw_html = response.text
    except httpx.TimeoutException:
        logger.warning(f"[web] fetch_page timeout: {url}")
        return {"success": False, "url": url, "content": f"Request timed out after {timeout}s."}
    except httpx.HTTPStatusError as e:
        logger.warning(f"[web] fetch_page HTTP {e.response.status_code}: {url}")
        return {"success": False, "url": url, "content": f"HTTP error {e.response.status_code}."}
    except httpx.RequestError as e:
        logger.warning(f"[web] fetch_page connection error: {e}")
        return {"success": False, "url": url, "content": f"Connection error: {e}"}
    except Exception as e:
        logger.exception(f"[web] fetch_page unexpected error: {url}")
        return {"success": False, "url": url, "content": str(e)}

    text = _strip_html(raw_html)

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
        logger.info(f"[web] fetch_page truncated to {max_chars} chars: {url}")
    else:
        logger.info(f"[web] fetch_page OK: {len(text)} chars from {url}")

    if truncated:
        text += f"\n\n[Content truncated at {max_chars} characters]"

    return {"success": True, "url": url, "content": text}


# ---------------------------------------------------------------------------
# Registration — call this once at startup from main.py
# ---------------------------------------------------------------------------

def register_web_tools() -> None:
    """Register search_web and fetch_page into the global tool registry."""

    register_tool(
        name="search_web",
        description=(
            "Search the web using SearXNG (self-hosted) with DuckDuckGo as fallback. "
            "No API key required. Returns a list of results with title, URL, and description."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=search_web,
        is_destructive=False,
    )

    register_tool(
        name="fetch_page",
        description=(
            "Fetch the plain-text content of a URL. "
            "Strips HTML, decodes entities, and truncates at 8000 characters. "
            "Use this to read a specific page after searching the web."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (must start with http:// or https://).",
                },
            },
            "required": ["url"],
        },
        handler=fetch_page,
        is_destructive=False,
    )
