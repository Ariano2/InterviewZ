"""
agents/github_publisher.py
GitHub Device Flow OAuth + repo publishing.

Device flow (no redirect URL needed):
  1. request_device_code(client_id) → user_code, device_code, verification_uri, interval
  2. User visits verification_uri and enters user_code in their browser
  3. poll_for_token(client_id, client_secret, device_code, interval) → token or error
  4. get_github_username(token) → username string
  5. publish_portfolio(token, files, repo_name) → (pages_url, error)
"""
import base64
import time
import requests
from typing import Dict, Tuple

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL       = "https://github.com/login/oauth/access_token"
GITHUB_API_BASE        = "https://api.github.com"
GITHUB_DEVICE_AUTH_URL = "https://github.com/login/device"


# ── Device Flow ───────────────────────────────────────────────────────────────

def request_device_code(client_id: str) -> Tuple[dict, str]:
    """
    Step 1 — request a device + user code from GitHub.
    Returns (data_dict, error_message).
    data_dict keys: device_code, user_code, verification_uri, expires_in, interval
    """
    resp = requests.post(
        GITHUB_DEVICE_CODE_URL,
        headers={"Accept": "application/json"},
        data={"client_id": client_id, "scope": "repo"},
        timeout=10,
    )
    if resp.status_code != 200:
        return {}, f"GitHub returned {resp.status_code}"
    data = resp.json()
    if "error" in data:
        return {}, data.get("error_description", data["error"])
    return data, ""


def poll_for_token(
    client_id: str,
    client_secret: str,
    device_code: str,
    interval: int = 5,
    timeout: int = 300,
) -> Tuple[str, str]:
    """
    Step 3 — poll until user authorises or timeout.
    Returns (access_token, error_message).
    Caller should run this in a background thread or use st.spinner loop.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        resp = requests.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id":     client_id,
                "client_secret": client_secret,
                "device_code":   device_code,
                "grant_type":    "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("access_token"):
            return data["access_token"], ""
        err = data.get("error", "")
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            interval += 5
            continue
        if err in ("expired_token", "access_denied"):
            return "", f"Auth failed: {err.replace('_', ' ')}. Please try again."
        # Unknown error — keep polling
    return "", "Timed out waiting for GitHub authorisation (5 min). Please try again."


# ── User info ─────────────────────────────────────────────────────────────────

def get_github_username(token: str) -> str:
    """Fetch the authenticated user's GitHub login."""
    resp = requests.get(
        f"{GITHUB_API_BASE}/user",
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
        timeout=10,
    )
    return resp.json().get("login", "")


# ── GitHub REST wrapper ───────────────────────────────────────────────────────

def _api(method: str, path: str, token: str, **kwargs):
    return requests.request(
        method,
        f"{GITHUB_API_BASE}{path}",
        headers={
            "Authorization":        f"token {token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=15,
        **kwargs,
    )


# ── Publishing ────────────────────────────────────────────────────────────────

def publish_portfolio(
    token: str,
    files: Dict[str, str],
    repo_name: str = "portfolio",
    description: str = "My personal portfolio — built with PrepSense AI",
) -> Tuple[str, str]:
    """
    Creates a GitHub repo and pushes files via the Contents API (simpler,
    no blob propagation race). Enables GitHub Pages on main branch.
    Returns (pages_url, error_message). pages_url is '' on hard failure.
    """
    username = get_github_username(token)
    if not username:
        return "", "Could not fetch GitHub username — token may have expired."

    # 1. Delete existing repo so re-runs are idempotent
    _api("DELETE", f"/repos/{username}/{repo_name}", token)
    time.sleep(2)  # GitHub needs a moment to fully drop the repo

    # 2. Create repo with auto_init=True so main branch exists immediately
    cr = _api("POST", "/user/repos", token, json={
        "name":        repo_name,
        "description": description,
        "private":     False,
        "auto_init":   True,   # creates initial commit + main branch
    })
    if cr.status_code not in (200, 201):
        return "", f"Could not create repo: {cr.json().get('message', cr.status_code)}"

    time.sleep(2)  # let GitHub fully initialise the repo before writing files

    # 3. Push each file via Contents API — simple, reliable, no blob race
    for filename, content in files.items():
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Check if file already exists (auto_init may have created README.md)
        get_r = _api("GET", f"/repos/{username}/{repo_name}/contents/{filename}", token)
        existing_sha = get_r.json().get("sha") if get_r.status_code == 200 else None

        payload = {
            "message": f"Add {filename}",
            "content": encoded,
        }
        if existing_sha:
            payload["sha"] = existing_sha  # required for updates

        put_r = _api("PUT", f"/repos/{username}/{repo_name}/contents/{filename}", token, json=payload)
        if put_r.status_code not in (200, 201):
            return "", f"Failed to push {filename}: {put_r.json().get('message', put_r.status_code)}"

    # 4. Enable GitHub Pages on main branch
    time.sleep(2)
    pages_r = _api("POST", f"/repos/{username}/{repo_name}/pages", token,
                   json={"source": {"branch": "main", "path": "/"}})

    pages_url = f"https://{username}.github.io/{repo_name}"
    if pages_r.status_code not in (200, 201, 409, 422):
        return pages_url, (
            f"Pages auto-enable failed ({pages_r.status_code}). "
            "Enable manually: repo Settings → Pages → Branch: main / root."
        )

    return pages_url, ""
