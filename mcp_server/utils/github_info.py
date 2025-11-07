import sys
import httpx
from typing import Any
from utils.constants import REPO_OWNER, REPO_NAME, BASE_URL, HEADERS


def get_pr_data(pr_number: int, github_client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetch the  main data from the PR

    Args:
        pr_number: The PR number
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The PR data
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}")
    response.raise_for_status()
    return response.json()


def get_pr_reviews(pr_number: int, github_client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetch the PR review states

    Args:
        pr_number: The PR number
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The PR reviews
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews")
    response.raise_for_status()
    return response.json()


def get_pr_files(pr_number: int, github_client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetch the PR files

    Args:
        pr_number: The PR number
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The PR files
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files")
    response.raise_for_status()
    return response.json()


def get_pr_comments(pr_number: int, github_client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetch PR comments

    Args:
        pr_number: The PR number
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The PR comments
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments")
    response.raise_for_status()
    return response.json()


def get_pr_diff(pr_number: int, github_client: httpx.Client | None = None) -> str:
    """Fetch the unified diff for a PR

    Args:
        pr_number: The PR number
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The PR diff
    """
    response = github_client.get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}",
        headers={"Accept": "application/vnd.github.v3.diff"},
    )
    response.raise_for_status()
    return response.text


def get_open_prs(
    state="open", sort="created", direction="desc", github_client: httpx.Client | None = None
) -> list[dict[str, Any]]:
    """Fetch all the open PRs

    Args:
        state: The state of the PR
        sort: The sort order
        direction: The direction of the sort
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The open PRs
    """
    prs = []
    page = 1
    while True:
        params = {"state": state, "sort": sort, "direction": direction, "page": page, "per_page": 100}
        response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls", params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        prs.extend(data)
        page += 1
    return prs


def compare_branches(base: str, head: str, github_client: httpx.Client | None = None) -> dict[str, Any]:
    """Compare two branches

    Args:
        base: The base branch
        head: The head branch
        github_client: Optional github client (uses module default if not provided)

    Returns:
        The comparison data
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/compare/{base}...{head}")
    response.raise_for_status()
    return response.json()


def get_ci_check_runs(commit_sha: str, github_client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetch CI check runs for a specific commit

    Args:
        commit_sha: The commit SHA to check
        github_client: Optional github client (uses module default if not provided)

    Returns:
        List of check runs for the commit
    """
    try:
        response = github_client.get(
            f"/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit_sha}/check-runs",
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("check_runs", [])
    except Exception as e:
        print(f"Warning: Could not fetch CI check runs: {e}", file=sys.stderr)
        return []
