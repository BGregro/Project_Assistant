# Remote Access Setup

## Find your auth token

Check the server console on first startup — it prints the token clearly:

```
============================================================
[remote] AUTH TOKEN GENERATED (first run):
[remote] <your-token-here>
[remote] Copy this token — you need it to connect remotely.
[remote] It is saved in config.json and won't change.
============================================================
```

Or open `config.json` and look for `remote_access.auth_token`.

---

## Access from the same WiFi network

1. Find your laptop's local IP: run `ipconfig` in cmd, look for **IPv4 Address**
2. On your phone browser: `http://192.168.x.x:8000/login`
3. Paste your auth token and press Connect

---

## Access from anywhere — Tailscale (recommended)

Tailscale is a zero-config VPN that works through NAT, firewalls, and mobile
data. All traffic is end-to-end encrypted (WireGuard).

1. Install Tailscale on your laptop: <https://tailscale.com/download>
2. Install Tailscale on your phone (iOS / Android app)
3. Sign in to the **same account** on both devices
4. Find your laptop's Tailscale IP in the Tailscale app (looks like `100.x.x.x`)
5. On your phone: `http://100.x.x.x:8000/login`

---

## Access from anywhere — Cloudflare Tunnel (alternative)

Good if you want a fixed public URL without installing a VPN.

1. Install cloudflared: <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>
2. Run: `cloudflared tunnel --url http://localhost:8000`
3. Cloudflare gives you a public `https://...trycloudflare.com` URL
4. Access via `/login` on that URL

---

## Security notes

- The auth token is a 32-byte random URL-safe string (~256 bits of entropy)
- It is never transmitted in plain text over Tailscale (encrypted VPN) or Cloudflare (HTTPS)
- Over plain local WiFi (`http://`) the token is sent unencrypted — use Tailscale
  or Cloudflare for any access outside your trusted home network
- **To regenerate the token:** delete the value of `auth_token` in `config.json`
  and restart the server — a new token is printed to the console
- **To disable auth** (local-only use): set `"require_auth": false` in `config.json`
- The server now binds to `0.0.0.0:8000`; make sure your OS firewall allows
  port 8000 if you intend to connect from another device

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/login` shows but Connect fails | Double-check the token — copy from config.json directly |
| WebSocket disconnects immediately | Token mismatch — verify the URL contains `?token=...` |
| Can't reach from phone on WiFi | Check Windows Defender Firewall: allow inbound port 8000 |
| Token printed once but now missing | Check `config.json` — token is saved there on first run |
