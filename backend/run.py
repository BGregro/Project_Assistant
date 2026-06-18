"""
run.py  —  Development server launcher

Replaces `uvicorn main:app --reload` on Windows because uvicorn's WatchFiles
reloader spawns a fresh worker process that ignores any event loop policy set
in the parent.  We work around this by:

  1. Setting WindowsProactorEventLoopPolicy globally (affects this process and
     any subprocess that inherits the environment).
  2. Passing loop="none" to uvicorn so it does NOT create its own loop before
     our app code runs.
  3. Explicitly creating and installing a ProactorEventLoop before handing
     control to uvicorn.Server.serve().

Playwright (and asyncio subprocesses in general, including execute_code) need
ProactorEventLoop on Windows to spawn child processes.  SelectorEventLoop —
uvicorn's default on Windows — raises NotImplementedError for subprocess calls.

Usage (from backend/ with venv active):
    python run.py
"""

import sys
import asyncio

# Step 1: set the policy process-wide so any new loop created anywhere uses Proactor
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn


def main() -> None:
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        # "none" tells uvicorn not to touch the event loop — we manage it below
        loop="none",
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        # Step 2: create a ProactorEventLoop explicitly and install it
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        # Non-Windows: let uvicorn handle the loop normally
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
