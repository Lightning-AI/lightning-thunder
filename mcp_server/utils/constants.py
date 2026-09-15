import os


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "Lightning-AI"
REPO_NAME = "lightning-thunder"
BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
