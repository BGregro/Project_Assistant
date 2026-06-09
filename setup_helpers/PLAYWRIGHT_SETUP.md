# Playwright Setup (Phase 3i)

Required for browser automation tools (`browser_open`, `browser_read`, `browser_screenshot`).

## Install

```
pip install playwright
playwright install chromium
```

## Verify

Start the agent and send:

> Open https://example.com and read its content using the browser.

Expected: agent calls `browser_open` then `browser_read` and returns the page text.

## Notes

- Chromium runs headless (no window appears)
- The browser instance persists across tool calls within one session — no re-launch overhead
- Closes automatically on server shutdown
- If not installed, browser tools are simply not registered — all other tools work normally
- `browser_screenshot` saves PNG files to the `outputs/` directory

## When to use browser tools vs fetch_page

| Situation | Recommended tool |
|---|---|
| Static HTML page (news article, docs, blog) | `fetch_page` — faster, no Chromium needed |
| JavaScript-rendered page (React/Vue SPA, dashboards) | `browser_open` → `browser_read` |
| Page returns empty or truncated content via `fetch_page` | Switch to `browser_open` → `browser_read` |
| Need a visual snapshot of what a page looks like | `browser_screenshot` |
