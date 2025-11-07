import httpx


def get_pr_data(github_client: httpx.Client, pr_number: int) -> dict[str, Any]:
    """Fetch the  main data from the PR

    Args:
        github_client: The github client
        pr_number: The PR number

    Returns:
        The PR data
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}")
    response.raise_for_status()
    return response.json()


def get_pr_reviews(github_client: httpx.Client, pr_number: int) -> list[dict[str, Any]]:
    """Fetch the PR review states

    Args:
        github_client: The github client
        pr_number: The PR number

    Returns:
        The PR reviews
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews")
    response.raise_for_status()
    return response.json()


def get_pr_files(github_client: httpx.Client, pr_number: int) -> list[dict[str, Any]]:
    """Fetch the PR files
    Args:
        github_client: The github client
        pr_number: The PR number

    Returns:
        The PR files
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files")
    response.raise_for_status()
    return response.json()


def get_pr_comments(github_client: httpx.Client, pr_number: int) -> list[dict[str, Any]]:
    """Fetch PR comments
    Args:
        github_client: The github client
        pr_number: The PR number

    Returns:
        The PR comments
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments")
    response.raise_for_status()
    return response.json()


def get_pr_diff(github_client: httpx.Client, pr_number: int) -> str:
    """Fetch the unified diff for a PR

    Args:
        github_client: The github client
        pr_number: The PR number

    Returns:
        The PR diff
    """
    response = github_client.get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}",
        headers={**HEADERS, "Accept": "application/vnd.github.v3.diff"},
    )
    response.raise_for_status()
    return response.text


def get_open_prs(github_client: httpx.Client, state="open", sort="created", direction="desc") -> list[dict[str, Any]]:
    """Fetch all the open PRs

    Args:
        github_client: The github client
        state: The state of the PR
        sort: The sort order
        direction: The direction of the sort

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


def compare_branches(github_client: httpx.Client, base: str, head: str) -> dict[str, Any]:
    """Compare two branches

    Args:
        github_client: The github client
        base: The base branch
        head: The head branch

    Returns:
        The comparison data
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/compare/{base}...{head}")
    response.raise_for_status()
    return response.json()
