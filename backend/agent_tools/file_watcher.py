"""
agent_tools/file_watcher.py  —  Phase 9c: File System Watcher Tools

Monitors directories for file-system events using watchdog.
When a file matching a glob pattern changes, the agent fires a task automatically.

Features:
    - Glob pattern filtering via fnmatch
    - {filename} and {filepath} template substitution in action messages
    - Per-file debounce: ignores repeat triggers within 5 seconds
    - Module-level singleton _manager shared by all tools
    - set_refs() wires in TaskRunner, AgentCore, and send_event at startup

Tools registered:
    watch_directory(watch_id, path, pattern, action_message)  — destructive
    stop_watching(watch_id)                                   — destructive
    list_watches()                                             — non-destructive

Requires: watchdog>=4.0.0 (add to requirements.txt)
"""

import asyncio
import logging
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Awaitable, Any

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import: watchdog is optional — tools return a helpful error if missing
# ---------------------------------------------------------------------------

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    Observer = None                 # type: ignore
    FileSystemEventHandler = object # type: ignore


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------

class _PatternEventHandler(FileSystemEventHandler):
    """
    A watchdog event handler that:
      1. Filters events by glob pattern (fnmatch)
      2. Debounces: ignores a second trigger on the same file within 5 seconds
      3. On match: substitutes {filename}/{filepath} in action_message and fires
         the agent as a background asyncio task.
    """

    DEBOUNCE_SECONDS = 5.0

    def __init__(
        self,
        watch_id: str,
        pattern: str,
        action_message: str,
        loop: asyncio.AbstractEventLoop,
        fire_fn: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.watch_id      = watch_id
        self.pattern       = pattern
        self.action_message = action_message
        self._loop         = loop
        self._fire_fn      = fire_fn
        # {filepath: last_trigger_time}
        self._last_seen: dict[str, float] = {}

    def _should_handle(self, src_path: str) -> bool:
        filename = Path(src_path).name
        if not fnmatch(filename, self.pattern):
            return False
        now = time.monotonic()
        last = self._last_seen.get(src_path, 0.0)
        if now - last < self.DEBOUNCE_SECONDS:
            logger.debug(
                f"[file_watcher:{self.watch_id}] Debounced: {filename} "
                f"({now - last:.1f}s < {self.DEBOUNCE_SECONDS}s)"
            )
            return False
        self._last_seen[src_path] = now
        return True

    def _dispatch_action(self, src_path: str) -> None:
        filename = Path(src_path).name
        message = (
            self.action_message
            .replace("{filename}", filename)
            .replace("{filepath}", src_path)
        )
        logger.info(
            f"[file_watcher:{self.watch_id}] Event → firing task: {message[:80]!r}"
        )
        # Thread-safe: schedule the async fire_fn in the main event loop
        self._loop.call_soon_threadsafe(self._fire_fn, message)

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._dispatch_action(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._dispatch_action(event.src_path)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class FileWatcherManager:
    """
    Manages a collection of watchdog observers, one per watch_directory() call.
    Supports starting, stopping, and listing active watchers.
    """

    def __init__(self) -> None:
        # {watch_id: {"observer": Observer, "path": str, "pattern": str, "action_message": str}}
        self._watchers: dict[str, dict] = {}

        # References set by main.py at startup
        self._task_runner_ref: Any  = None
        self._agent_ref:       Any  = None
        self._send_event_fn:   Callable | None = None
        self._loop:            asyncio.AbstractEventLoop | None = None

    def set_refs(
        self,
        task_runner: Any,
        agent: Any,
        send_event: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        """
        Wire in the TaskRunner, AgentCore, and broadcast send_event.
        Must be called from main.py after both objects are created.
        """
        self._task_runner_ref = task_runner
        self._agent_ref       = agent
        self._send_event_fn   = send_event
        # Capture the current event loop so threads can schedule on it
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None
        logger.info("[file_watcher] Refs set.")

    def _make_fire_fn(self, message: str) -> None:
        """
        Called from the watchdog thread via loop.call_soon_threadsafe.
        Schedules the agent task on the async event loop.
        """
        # We receive just the resolved message string here via closure
        pass  # Placeholder — see _create_fire_fn

    def _create_fire_fn(self) -> Callable[[str], None]:
        """
        Create a thread-safe callable that launches an agent task for the
        given message string.  Captures self so it has access to the refs.
        """
        manager = self

        def _fire(message: str) -> None:
            loop = manager._loop
            if loop is None or not loop.is_running():
                logger.warning("[file_watcher] Event loop not available — ignoring trigger.")
                return

            async def _run_task():
                if manager._send_event_fn is None or manager._task_runner_ref is None:
                    logger.warning("[file_watcher] Refs not set — cannot fire task.")
                    return
                try:
                    await manager._send_event_fn(
                        "status",
                        {"text": f"File watcher triggered: {message[:60]}"}
                    )
                    # Re-use the task runner's run_task mechanism via agent.run_with_task_runner
                    await manager._agent_ref.run_with_task_runner(
                        task_runner=manager._task_runner_ref,
                        user_message=message,
                        history=[],
                        send_event=manager._send_event_fn,
                        pending_confirmations={},
                        context_summary="[Triggered by file watcher]",
                    )
                except Exception as e:
                    logger.warning(f"[file_watcher] Task fire failed: {e}")

            asyncio.run_coroutine_threadsafe(_run_task(), loop)

        return _fire

    def watch(
        self,
        watch_id: str,
        path: str,
        pattern: str = "*",
        action_message: str = "",
    ) -> dict:
        """
        Start watching a directory.  Returns success/error dict.
        """
        if not _WATCHDOG_AVAILABLE:
            return {
                "success": False,
                "error": (
                    "watchdog is not installed. "
                    "Run: pip install watchdog>=4.0.0 --break-system-packages"
                ),
            }

        if watch_id in self._watchers:
            return {
                "success": False,
                "error": f"Watch ID '{watch_id}' is already active. Stop it first.",
            }

        watch_path = Path(path)
        if not watch_path.exists() or not watch_path.is_dir():
            return {
                "success": False,
                "error": f"Directory does not exist: {path}",
            }

        if not self._loop:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                return {
                    "success": False,
                    "error": "Event loop not available — call set_refs() first.",
                }

        fire_fn = self._create_fire_fn()
        handler  = _PatternEventHandler(
            watch_id=watch_id,
            pattern=pattern,
            action_message=action_message,
            loop=self._loop,
            fire_fn=fire_fn,
        )

        observer = Observer()
        observer.schedule(handler, str(watch_path), recursive=True)
        observer.start()

        self._watchers[watch_id] = {
            "observer":       observer,
            "path":           str(watch_path),
            "pattern":        pattern,
            "action_message": action_message,
        }

        logger.info(
            f"[file_watcher] Started watch '{watch_id}': "
            f"path={watch_path}, pattern={pattern!r}"
        )
        return {
            "success":        True,
            "watch_id":       watch_id,
            "path":           str(watch_path),
            "pattern":        pattern,
            "action_message": action_message,
        }

    def stop_watch(self, watch_id: str) -> dict:
        """Stop and remove a watcher by ID."""
        if watch_id not in self._watchers:
            return {
                "success": False,
                "error": f"No active watcher with ID '{watch_id}'.",
            }

        info = self._watchers.pop(watch_id)
        observer: Observer = info["observer"]
        try:
            observer.stop()
            observer.join(timeout=5)
        except Exception as e:
            logger.warning(f"[file_watcher] Error stopping observer '{watch_id}': {e}")

        logger.info(f"[file_watcher] Stopped watch '{watch_id}'.")
        return {"success": True, "watch_id": watch_id}

    def list_watches(self) -> list[dict]:
        """Return a list of all active watchers."""
        return [
            {
                "watch_id":       wid,
                "path":           info["path"],
                "pattern":        info["pattern"],
                "action_message": info["action_message"],
            }
            for wid, info in self._watchers.items()
        ]

    def shutdown(self) -> None:
        """Stop all active observers.  Called on server shutdown."""
        for watch_id in list(self._watchers.keys()):
            self.stop_watch(watch_id)
        logger.info("[file_watcher] All watchers stopped.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager = FileWatcherManager()


# ---------------------------------------------------------------------------
# Tool implementations (thin wrappers around the manager)
# ---------------------------------------------------------------------------

async def watch_directory(
    watch_id: str,
    path: str,
    pattern: str = "*",
    action_message: str = "",
) -> dict:
    """
    Start watching a directory for file changes.

    When a file matching `pattern` is created or modified, the agent fires
    `action_message` as a task.  Use {filename} and {filepath} as placeholders.
    """
    return _manager.watch(watch_id, path, pattern, action_message)


async def stop_watching(watch_id: str) -> dict:
    """Stop a directory watcher by its ID."""
    return _manager.stop_watch(watch_id)


async def list_watches() -> dict:
    """List all currently active file watchers."""
    watches = _manager.list_watches()
    return {
        "success": True,
        "watches": watches,
        "count":   len(watches),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_file_watcher_tools() -> None:
    """Register file watcher tools into the agent tool registry."""

    register_tool(
        name="watch_directory",
        description=(
            "Start watching a directory for file-system changes. "
            "When a file matching the glob pattern is created or modified, "
            "the agent automatically runs action_message as a task. "
            "Use {filename} and {filepath} as template placeholders in action_message. "
            "Example: watch_id='new_uploads', path='outputs/', pattern='*.csv', "
            "action_message='A new CSV was uploaded: {filename}. Summarise it.' "
            "Requires watchdog>=4.0.0."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "watch_id": {
                    "type": "string",
                    "description": "Unique name for this watcher (e.g. 'uploads_watch').",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the directory to watch.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for filenames to match, e.g. '*.py', '*.csv', '*'. Default: '*'.",
                    "default": "*",
                },
                "action_message": {
                    "type": "string",
                    "description": (
                        "Task message to fire when a matching file event occurs. "
                        "Use {filename} and {filepath} as placeholders. "
                        "Example: 'New file detected: {filename}. Process it.'"
                    ),
                    "default": "",
                },
            },
            "required": ["watch_id", "path"],
        },
        handler=watch_directory,
        destructive=True,
    )

    register_tool(
        name="stop_watching",
        description=(
            "Stop a directory watcher by its watch_id. "
            "Use list_watches() to see active watcher IDs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "watch_id": {
                    "type": "string",
                    "description": "ID of the watcher to stop.",
                },
            },
            "required": ["watch_id"],
        },
        handler=stop_watching,
        destructive=True,
    )

    register_tool(
        name="list_watches",
        description=(
            "List all currently active directory watchers, "
            "showing their IDs, paths, patterns, and action messages."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=list_watches,
        destructive=False,
    )

    logger.info("[file_watcher] Registered tools: watch_directory, stop_watching, list_watches")
