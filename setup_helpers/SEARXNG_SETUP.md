# SearXNG Setup — Windows (Docker Desktop)

SearXNG is a self-hosted, privacy-respecting meta-search engine.
The agent uses it as the primary backend for the `search_web` tool,
with DuckDuckGo HTML scraping as an automatic fallback if SearXNG is offline.

---

## Prerequisites

**Docker Desktop for Windows**
Download and install from: https://www.docker.com/products/docker-desktop

After installation, start Docker Desktop and wait for the whale icon in the
system tray to show "Docker Desktop is running".

---

## Step 1 — Pull and run SearXNG

Open PowerShell or Command Prompt and run:

```
docker run -d --name searxng -p 8888:8080 searxng/searxng
```

What each flag does:
- `-d`              — run in the background (detached mode)
- `--name searxng`  — give the container a memorable name
- `-p 8888:8080`    — map localhost:8888 → container port 8080
- `searxng/searxng` — official SearXNG image from Docker Hub

---

## Step 2 — Verify it works

Open your browser and go to: http://localhost:8888

You should see the SearXNG search page. Try a search to confirm it works.

---

## Step 3 — Enable JSON output (required for the agent)

By default SearXNG may have the JSON format disabled. To enable it:

```
docker exec searxng sed -i 's/- format: json/#- format: json/' /etc/searxng/settings.yml
```

Then apply the correct setting:

```
docker exec searxng sh -c "grep -q 'json' /etc/searxng/settings.yml || echo '  - format: json' >> /etc/searxng/settings.yml"
```

Restart the container to pick up the change:

```
docker restart searxng
```

Verify JSON works by visiting: http://localhost:8888/search?q=test&format=json
You should see a JSON response, not an HTML page.

---

## After a reboot

Docker containers stop when Windows shuts down. To start SearXNG again:

```
docker start searxng
```

To check if it is currently running:

```
docker ps
```

---

## Changing the port

The default port **8888** matches the value in `config.json`:

```json
"tools": {
  "web": {
    "searxng_url": "http://localhost:8888"
  }
}
```

If you need a different port (e.g. 8080 is already taken), change both:

1. The `-p` flag when creating the container:
   ```
   docker run -d --name searxng -p 9999:8080 searxng/searxng
   ```

2. The `searxng_url` value in `config.json`:
   ```json
   "searxng_url": "http://localhost:9999"
   ```

---

## Removing the container

If you want to stop and remove SearXNG entirely:

```
docker stop searxng
docker rm searxng
```

---

## Fallback behaviour

If SearXNG is offline or unreachable, the `search_web` tool automatically
falls back to scraping DuckDuckGo's HTML interface — no action needed.
The fallback is logged as a warning so you can see when it is being used:

```
WARNING  [web] SearXNG unavailable (http://localhost:8888): ...
INFO     [web] Falling back to DuckDuckGo HTML scrape.
```
