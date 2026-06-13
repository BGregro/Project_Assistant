from __future__ import annotations

"""
youtube_tool.py  —  Phase 5c: YouTube Data API integration

All tools are read-only (no upload capability yet).  An API key stored via
the credential manager is required — every tool checks for it at call time
so the file can be imported safely even before a key is stored.

Endpoints used:
    GET /search            → youtube_search
    GET /videos            → youtube_get_video_stats, youtube_get_trending
    GET /commentThreads    → youtube_get_video_comments
    GET /channels          → youtube_get_channel_info
    GET /captions          → youtube_search_captions

ISO 8601 duration examples:
    PT4S      →  "4s"
    PT1M30S   →  "1m 30s"
    PT1H2M3S  →  "1h 2m 3s"
"""

import logging
import re

import httpx

from agent_tools import register_tool

logger = logging.getLogger(__name__)

_YT_BASE = "https://www.googleapis.com/youtube/v3"

# ---------------------------------------------------------------------------
# Credential helper — wrapped in try/except so this module loads even when
# credentials.py hasn't been registered yet (e.g. during unit tests or
# partial startups).
# ---------------------------------------------------------------------------

try:
    from agent_tools.credentials import get_credential as _get_credential_fn
    _CREDENTIALS_AVAILABLE = True
except ImportError:
    _get_credential_fn = None  # type: ignore
    _CREDENTIALS_AVAILABLE = False

_NO_KEY_ERROR = {
    "success": False,
    "error": (
        "YouTube API key not stored. "
        "Call store_credential('youtube_api_key', 'your_key') first. "
        "Get a free key at https://console.developers.google.com"
    ),
}


async def _get_api_key() -> str | None:
    """
    Retrieve the stored YouTube API key via the credential manager.

    Returns the plaintext key string on success, or None if the key is
    missing, the credential module is unavailable, or decryption fails.
    """
    if not _CREDENTIALS_AVAILABLE or _get_credential_fn is None:
        return None
    result = await _get_credential_fn("youtube_api_key")
    if result.get("success"):
        return result.get("value")
    return None


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

async def _yt_get(endpoint: str, params: dict) -> tuple[bool, dict]:
    """
    Execute a GET request against the YouTube Data API v3.

    Appends the `key` query parameter automatically.  The caller must supply
    all other parameters (part, q, id, etc.) in `params`.

    Returns:
        (True,  response_json)                 on any 2xx response
        (False, {"error": status, "message": …}) on any non-2xx response
    """
    api_key = await _get_api_key()
    if api_key is None:
        return False, {"error": "no_key", "message": "YouTube API key not stored."}

    url = f"{_YT_BASE}/{endpoint.lstrip('/')}"
    params = {**params, "key": api_key}

    logger.debug(f"[youtube] GET {url} params={list(params.keys())}")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as exc:
        logger.warning(f"[youtube] Request error: {exc}")
        return False, {"error": "request_error", "message": str(exc)}

    try:
        body = resp.json()
    except Exception:
        body = {}

    if resp.is_success:
        return True, body

    message = body.get("error", {}).get("message", "Unknown error")
    logger.warning(f"[youtube] API error {resp.status_code}: {message}")
    return False, {"error": resp.status_code, "message": message}


# ---------------------------------------------------------------------------
# ISO 8601 duration parser
# ---------------------------------------------------------------------------

def _parse_duration(iso: str) -> str:
    """
    Convert an ISO 8601 duration string to a human-readable format.

    Examples:
        "PT4S"      → "4s"
        "PT1M30S"   → "1m 30s"
        "PT1H2M3S"  → "1h 2m 3s"
        "P1DT2H"    → "1d 2h"

    Unknown / malformed values are returned as-is.
    """
    if not iso or not iso.startswith("P"):
        return iso or ""

    pattern = re.compile(
        r"P(?:(\d+)D)?"        # days
        r"(?:T"
        r"(?:(\d+)H)?"         # hours
        r"(?:(\d+)M)?"         # minutes
        r"(?:(\d+)S)?"         # seconds
        r")?"
    )
    m = pattern.fullmatch(iso)
    if not m:
        return iso

    days, hours, minutes, seconds = m.groups()
    parts = []
    if days:    parts.append(f"{days}d")
    if hours:   parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    if seconds: parts.append(f"{seconds}s")
    return " ".join(parts) if parts else "0s"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def youtube_search(
    query: str,
    max_results: int = 10,
    video_type: str = "video",
) -> dict:
    """
    Search YouTube for videos (or channels/playlists) matching a query.

    Args:
        query:       Search terms.
        max_results: Number of results to return (1–50, default 10).
        video_type:  "video" (default), "channel", or "playlist".

    Returns a list of result items with video_id, title, description,
    channel, published date, and thumbnail URL.
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("search", {
        "part":       "snippet",
        "q":          query,
        "type":       video_type,
        "maxResults": min(max_results, 50),
    })

    if not ok:
        return {"success": False, "error": data.get("message", "API error")}

    items = data.get("items", [])
    results = []
    for item in items:
        snippet = item.get("snippet", {})
        # The id field structure differs between video / channel / playlist
        id_block = item.get("id", {})
        video_id = (
            id_block.get("videoId")
            or id_block.get("channelId")
            or id_block.get("playlistId")
            or ""
        )
        results.append({
            "video_id":      video_id,
            "title":         snippet.get("title", ""),
            "description":   snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_at":  snippet.get("publishedAt", ""),
            "thumbnail_url": (
                snippet.get("thumbnails", {})
                       .get("medium", {})
                       .get("url", "")
            ),
        })

    return {
        "success":     True,
        "query":       query,
        "total_found": len(results),
        "results":     results,
    }


async def youtube_get_video_stats(video_id: str) -> dict:
    """
    Return detailed statistics and metadata for a single YouTube video.

    Retrieves views, likes, comment count, duration (human-readable),
    tags, channel name, and publication date.

    Args:
        video_id: The YouTube video ID (the part after ?v= in the URL).
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("videos", {
        "part": "statistics,snippet,contentDetails",
        "id":   video_id,
    })

    if not ok:
        return {"success": False, "error": data.get("message", "API error")}

    items = data.get("items", [])
    if not items:
        return {"success": False, "error": f"Video '{video_id}' not found."}

    item      = items[0]
    snippet   = item.get("snippet", {})
    stats     = item.get("statistics", {})
    details   = item.get("contentDetails", {})

    return {
        "success":     True,
        "video_id":    video_id,
        "title":       snippet.get("title", ""),
        "channel":     snippet.get("channelTitle", ""),
        "views":       int(stats.get("viewCount", 0)),
        "likes":       int(stats.get("likeCount", 0)),
        "comments":    int(stats.get("commentCount", 0)),
        "duration":    _parse_duration(details.get("duration", "")),
        "tags":        snippet.get("tags", []),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
    }


async def youtube_get_trending(
    region_code: str = "US",
    category_id: str = "0",
    max_results: int = 20,
) -> dict:
    """
    Return the current trending / most-popular videos for a given region
    and optional video category.

    Args:
        region_code: ISO 3166-1 alpha-2 country code (e.g. "US", "GB", "JP").
        category_id: YouTube video category ID string.
                     "0" = all categories (default).
                     Common IDs: "10" music, "20" gaming, "22" people/blogs,
                     "24" entertainment, "28" science/tech.
        max_results: Number of videos to return (1–50, default 20).

    Returns a ranked list; rank 1 is most popular.
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("videos", {
        "part":            "snippet,statistics",
        "chart":           "mostPopular",
        "regionCode":      region_code,
        "videoCategoryId": category_id,
        "maxResults":      min(max_results, 50),
    })

    if not ok:
        return {"success": False, "error": data.get("message", "API error")}

    items = data.get("items", [])
    results = []
    for rank, item in enumerate(items, start=1):
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        results.append({
            "rank":          rank,
            "video_id":      item.get("id", ""),
            "title":         snippet.get("title", ""),
            "channel":       snippet.get("channelTitle", ""),
            "views":         int(stats.get("viewCount", 0)),
            "likes":         int(stats.get("likeCount", 0)),
            "published_at":  snippet.get("publishedAt", ""),
            "thumbnail_url": (
                snippet.get("thumbnails", {})
                       .get("medium", {})
                       .get("url", "")
            ),
        })

    return {
        "success":     True,
        "region_code": region_code,
        "category_id": category_id,
        "count":       len(results),
        "trending":    results,
    }


async def youtube_get_video_comments(
    video_id: str,
    max_results: int = 20,
) -> dict:
    """
    Return the top comments on a YouTube video, ordered by relevance.

    Comment text is truncated to 200 characters.  Only top-level comments
    are returned (not replies to comments).

    Args:
        video_id:    The YouTube video ID.
        max_results: Number of comments to return (1–100, default 20).
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("commentThreads", {
        "part":       "snippet",
        "videoId":    video_id,
        "order":      "relevance",
        "maxResults": min(max_results, 100),
    })

    if not ok:
        # Comments are disabled on some videos — surface a friendly message.
        msg = data.get("message", "API error")
        if "disabled" in msg.lower() or "403" in str(data.get("error", "")):
            return {
                "success": False,
                "error":   "Comments are disabled for this video.",
            }
        return {"success": False, "error": msg}

    items = data.get("items", [])
    comments = []
    for item in items:
        top = item.get("snippet", {}).get("topLevelComment", {})
        snip = top.get("snippet", {})
        text = snip.get("textDisplay", "")
        comments.append({
            "author":       snip.get("authorDisplayName", ""),
            "text":         text[:200] + ("…" if len(text) > 200 else ""),
            "likes":        int(snip.get("likeCount", 0)),
            "published_at": snip.get("publishedAt", ""),
        })

    return {
        "success":   True,
        "video_id":  video_id,
        "count":     len(comments),
        "comments":  comments,
    }


async def youtube_get_channel_info(channel_id: str) -> dict:
    """
    Return analytics and metadata for a YouTube channel.

    Args:
        channel_id: The channel ID (starts with "UC…") or a custom handle.
                    To find a channel ID from a handle, search for the
                    channel first with youtube_search(query, video_type="channel").
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("channels", {
        "part": "statistics,snippet,brandingSettings",
        "id":   channel_id,
    })

    if not ok:
        return {"success": False, "error": data.get("message", "API error")}

    items = data.get("items", [])
    if not items:
        return {"success": False, "error": f"Channel '{channel_id}' not found."}

    item     = items[0]
    snippet  = item.get("snippet", {})
    stats    = item.get("statistics", {})
    branding = item.get("brandingSettings", {}).get("channel", {})

    return {
        "success":     True,
        "channel_id":  channel_id,
        "name":        snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "custom_url":  snippet.get("customUrl", ""),
        "country":     snippet.get("country", ""),
        "published_at": snippet.get("publishedAt", ""),
        "subscribers": int(stats.get("subscriberCount", 0)),
        "total_views": int(stats.get("viewCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "keywords":    branding.get("keywords", ""),
    }


async def youtube_search_captions(video_id: str) -> dict:
    """
    List the caption tracks available for a YouTube video.

    NOTE: This tool lists caption track metadata only.  Downloading the
    actual caption content requires OAuth 2.0 user authentication, which is
    not yet implemented.  Use a third-party transcript service (e.g.
    youtube-transcript-api Python library) for caption text extraction.

    Args:
        video_id: The YouTube video ID.

    Returns a list of caption tracks with language, name, kind, and
    whether the track is auto-generated or community-contributed (CC).
    """
    api_key = await _get_api_key()
    if api_key is None:
        return _NO_KEY_ERROR

    ok, data = await _yt_get("captions", {
        "part":    "snippet",
        "videoId": video_id,
    })

    if not ok:
        return {"success": False, "error": data.get("message", "API error")}

    items = data.get("items", [])
    tracks = []
    for item in items:
        snip = item.get("snippet", {})
        tracks.append({
            "id":         item.get("id", ""),
            "language":   snip.get("language", ""),
            "name":       snip.get("name", ""),
            "track_kind": snip.get("trackKind", ""),      # "standard", "asr", "forced"
            "is_cc":      snip.get("trackKind") != "asr", # asr = auto-generated, not CC
        })

    return {
        "success":  True,
        "video_id": video_id,
        "count":    len(tracks),
        "tracks":   tracks,
        "note": (
            "Downloading caption content requires OAuth 2.0. "
            "For transcript extraction without OAuth, use the "
            "youtube-transcript-api Python package."
        ),
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_youtube_tools() -> None:
    """Register all six YouTube Data API tools with the tool registry."""

    register_tool(
        name="youtube_search",
        description=(
            "Search YouTube for videos, channels, or playlists. "
            "Returns video IDs, titles, descriptions, channel names, "
            "publication dates, and thumbnail URLs. "
            "Requires 'youtube_api_key' stored via store_credential. "
            "Use this to find videos on any topic, research competitors, "
            "or discover content for analysis. "
            "Set video_type to 'channel' to find channels by name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–50, default 10).",
                },
                "video_type": {
                    "type": "string",
                    "description": "Type of result: 'video' (default), 'channel', or 'playlist'.",
                    "enum": ["video", "channel", "playlist"],
                },
            },
            "required": ["query"],
        },
        handler=youtube_search,
        destructive=False,
    )

    register_tool(
        name="youtube_get_video_stats",
        description=(
            "Get detailed statistics and metadata for a single YouTube video. "
            "Returns views, likes, comment count, duration (human-readable), "
            "tags, channel name, and publication date. "
            "Requires 'youtube_api_key' stored via store_credential. "
            "Use this to analyse the performance of a specific video."
        ),
        parameters={
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID (the part after ?v= in the URL).",
                },
            },
            "required": ["video_id"],
        },
        handler=youtube_get_video_stats,
        destructive=False,
    )

    register_tool(
        name="youtube_get_trending",
        description=(
            "Return the current trending / most-popular videos for a country and category. "
            "Results are ranked 1-N (rank 1 = most popular). "
            "Useful for content strategy research and trend analysis. "
            "Requires 'youtube_api_key' stored via store_credential. "
            "Common category_id values: '0' all, '10' music, '20' gaming, "
            "'22' people/blogs, '24' entertainment, '28' science/tech."
        ),
        parameters={
            "type": "object",
            "properties": {
                "region_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB', 'JP'). Default: 'US'.",
                },
                "category_id": {
                    "type": "string",
                    "description": "YouTube category ID string. '0' = all categories (default).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of videos to return (1–50, default 20).",
                },
            },
            "required": [],
        },
        handler=youtube_get_trending,
        destructive=False,
    )

    register_tool(
        name="youtube_get_video_comments",
        description=(
            "Return the top comments on a YouTube video, ordered by relevance. "
            "Comment text is truncated to 200 characters. "
            "Returns author name, comment text, like count, and date. "
            "Requires 'youtube_api_key' stored via store_credential. "
            "Useful for audience sentiment analysis and content research. "
            "Note: some videos have comments disabled — the tool returns a clear error in that case."
        ),
        parameters={
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of comments to return (1–100, default 20).",
                },
            },
            "required": ["video_id"],
        },
        handler=youtube_get_video_comments,
        destructive=False,
    )

    register_tool(
        name="youtube_get_channel_info",
        description=(
            "Return analytics and metadata for a YouTube channel. "
            "Returns subscriber count, total views, video count, country, "
            "custom URL, and channel description. "
            "Requires 'youtube_api_key' stored via store_credential. "
            "To find a channel_id from a handle, first call "
            "youtube_search(query, video_type='channel') and use the video_id "
            "field from the result (channel IDs start with 'UC')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID (starts with 'UC…'). Find via youtube_search with video_type='channel'.",
                },
            },
            "required": ["channel_id"],
        },
        handler=youtube_get_channel_info,
        destructive=False,
    )

    register_tool(
        name="youtube_search_captions",
        description=(
            "List the caption tracks available for a YouTube video. "
            "Returns language, track name, kind (standard/asr/forced), and "
            "whether it is a community-contributed closed-caption track. "
            "NOTE: This tool only lists available tracks — downloading caption "
            "content requires OAuth 2.0 authentication (not yet implemented). "
            "For full transcript extraction, use the youtube-transcript-api "
            "Python package via execute_code instead. "
            "Requires 'youtube_api_key' stored via store_credential."
        ),
        parameters={
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID.",
                },
            },
            "required": ["video_id"],
        },
        handler=youtube_search_captions,
        destructive=False,
    )

    logger.info(
        "[youtube] Registered tools: youtube_search, youtube_get_video_stats, "
        "youtube_get_trending, youtube_get_video_comments, "
        "youtube_get_channel_info, youtube_search_captions"
    )
