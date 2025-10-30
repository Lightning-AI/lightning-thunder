import os
import sys
from datetime import datetime, timezone
from typing import Any
from dataclasses import dataclass, asdict
import json
import httpx
from mcp.server.fastmcp import FastMCP


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "Lightning-AI"
REPO_NAME = "lightning-thunder"
BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
# create the github client
github_client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
# initialize the mcp server
mcp = FastMCP("thunder-pr-inspector")


# Define all the data classes for the PR judgment
@dataclass
class RiskReasoning:
    breaking_changes: str
    security: str
    urgency: str


@dataclass
class RiskScore:
    breaking_changes: int  # 0-10 range
    security: int  # 0-10
    urgency_if_not_merged: int  # 0-10
    overall: int  # 0-10
    reasoning: RiskReasoning


@dataclass
class StalenessInfo:
    days_open: int
    days_since_update: int
    is_mergeable: bool | None
    has_conflicts: bool
    commits_behind_base: int | None


@dataclass
class ReviewStatus:
    total_reviews: int
    approved_reviews: int
    changes_requested: int
    pending_reviews: int


@dataclass
class PRAnalysis:
    number: int
    title: str
    author: str
    created_at: str  # change to datetime?
    updated_at: str  # change to datetime?
    url: str
    labels: list[str]
    summary: str
    risk_score: RiskScore
    priority_score: int
    staleness: StalenessInfo
    review_status: ReviewStatus
    files_changed: int
    additions: int
    deletions: int
    llm_summary: str
    llm_risk_assessment: str


# Define the github helpers
def get_pr_data(pr_number: int) -> dict[str, Any]:
    """Fetch the  main data from the PR"""
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}")
    response.raise_for_status()
    return response.json()


def get_pr_reviews(pr_number: int) -> list[dict[str, Any]]:
    """Fetch the PR review states"""
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews")
    response.raise_for_status()
    return response.json()


def get_pr_files(pr_number: int) -> list[dict[str, Any]]:
    """Fetch the PR files
    TODO this dpeends on pagination
    """
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files")
    response.raise_for_status()
    return response.json()


def get_pr_comments(pr_number: int) -> list[dict[str, Any]]:
    """Fetch PR comments"""
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments")
    response.raise_for_status()
    return response.json()


def get_pr_diff(pr_number: int) -> str:
    """Fetch the unified diff for a PR."""
    response = github_client.get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}",
        headers={**HEADERS, "Accept": "application/vnd.github.v3.diff"},
    )
    response.raise_for_status()
    return response.text


def get_open_prs(state="open", sort="created", direction="desc") -> list[dict[str, Any]]:
    """Fetch all the open PRs"""
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


def compare_branches(base: str, head: str) -> dict[str, Any]:
    response = github_client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/compare/{base}...{head}")
    response.raise_for_status()
    return response.json()


# HELPER FUNCTION
def calculate_days_diff(date_str: str) -> int:
    """Function to calculate the days difference between two dates"""
    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - date).days


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to dict recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


# Heuristic Analysis Engine
def assess_risk(
    pr: dict[str, Any],
    files: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    staleness: StalenessInfo,
) -> RiskScore:
    """Assess multi-dimensional risk for a PR."""

    breaking_changes_score = 0
    security_score = 0
    urgency_score = 0

    # Get PR text content
    title_and_body = (pr["title"] + " " + (pr["body"] or "")).lower()
    labels_lower = [label["name"].lower() for label in pr["labels"]]

    # === Breaking Changes Risk ===
    breaking_keywords = ["breaking", "deprecat", "remov", "incompatib", "major"]
    api_files = [f for f in files if any(x in f["filename"].lower() for x in ["api", "interface", "__init__.py"])]

    has_breaking_keywords = any(kw in title_and_body for kw in breaking_keywords)

    if has_breaking_keywords:
        breaking_changes_score += 5
    if len(api_files) > 0:
        breaking_changes_score += 2
    if len(files) > 20:
        breaking_changes_score += 2
    if any("breaking" in label for label in labels_lower):
        breaking_changes_score += 5

    breaking_changes_score = min(10, breaking_changes_score)

    # === Security Risk ===
    security_keywords = ["security", "vulnerability", "cve", "exploit", "auth", "credential", "token", "password"]
    security_files = [f for f in files if any(x in f["filename"].lower() for x in ["auth", "security", "credential"])]

    has_security_keywords = any(kw in title_and_body for kw in security_keywords)

    if has_security_keywords:
        security_score += 7
    if len(security_files) > 0:
        security_score += 3
    if any("security" in label for label in labels_lower):
        security_score += 8

    security_score = min(10, security_score)

    # === Urgency If Not Merged ===
    urgency_keywords = ["block", "critical", "urgent", "hotfix", "bug", "regression", "crash", "broken"]
    has_urgency_keywords = any(kw in title_and_body for kw in urgency_keywords)

    if has_urgency_keywords:
        urgency_score += 4
    if staleness.days_open > 90:
        urgency_score += 3  # Old PRs might be important
    if any(label in labels_lower for label in ["bug", "critical", "blocker"]):
        urgency_score += 5
    if len(comments) > 15:
        urgency_score += 2  # High engagement

    urgency_score = min(10, urgency_score)

    # Overall risk
    overall = round((breaking_changes_score + security_score + urgency_score) / 3)

    # Generate reasoning
    reasoning = RiskReasoning(
        breaking_changes=_generate_breaking_reasoning(
            breaking_changes_score, has_breaking_keywords, len(api_files), len(files)
        ),
        security=_generate_security_reasoning(security_score, has_security_keywords, len(security_files)),
        urgency=_generate_urgency_reasoning(urgency_score, has_urgency_keywords, staleness.days_open, len(comments)),
    )

    return RiskScore(
        breaking_changes=breaking_changes_score,
        security=security_score,
        urgency_if_not_merged=urgency_score,
        overall=overall,
        reasoning=reasoning,
    )


def _generate_breaking_reasoning(score: int, has_keywords: bool, api_files: int, total_files: int) -> str:
    # (Your original helper)
    reasons = []
    if has_keywords:
        reasons.append("contains breaking-related keywords")
    if api_files > 0:
        reasons.append(f"modifies {api_files} API/interface file(s)")
    if total_files > 20:
        reasons.append(f"large changeset ({total_files} files)")
    return (
        f"Score {score}/10: {', '.join(reasons)}"
        if reasons
        else f"Score {score}/10: no obvious breaking changes detected"
    )


def _generate_security_reasoning(score: int, has_keywords: bool, security_files: int) -> str:
    # (Your original helper)
    reasons = []
    if has_keywords:
        reasons.append("contains security-related keywords")
    if security_files > 0:
        reasons.append(f"modifies {security_files} security-related file(s)")
    return f"Score {score}/10: {', '.join(reasons)}" if reasons else f"Score {score}/10: no obvious security concerns"


def _generate_urgency_reasoning(score: int, has_keywords: bool, days_open: int, comment_count: int) -> str:
    # (Your original helper)
    reasons = []
    if has_keywords:
        reasons.append("contains urgent/critical keywords")
    if days_open > 90:
        reasons.append(f"open for {days_open} days")
    if comment_count > 15:
        reasons.append(f"high engagement ({comment_count} comments)")
    return f"Score {score}/10: {', '.join(reasons)}" if reasons else f"Score {score}/10: no urgent indicators"


def generate_summary_heuristic(
    pr: dict[str, Any], files: list[dict[str, Any]], comments: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> str:
    """Generate a human-readable summary of the PR based on heuristics."""
    # (This is your original generate_summary function)
    parts = [
        f"**{pr['title']}** by @{pr['user']['login']}",
        f"\n\n**Changes:** {len(files)} files modified, +{pr['additions']} -{pr['deletions']} lines",
    ]

    if pr["body"]:
        body_summary = pr["body"][:200].strip() + ("..." if len(pr["body"]) > 200 else "")
        parts.append(f"\n\n**Description:** {body_summary}")

    if comments or reviews:
        parts.append(f"\n\n**Activity:** {len(reviews)} reviews, {len(comments)} comments")

    return "".join(parts)


def calculate_priority(
    pr: dict[str, Any], risk: RiskScore, staleness: StalenessInfo, review_status: ReviewStatus
) -> int:
    """Calculate priority score (0-100) based on multiple factors."""
    # (This is your original priority function)
    score = 0.0
    score += risk.security * 2
    score += risk.urgency_if_not_merged * 1.5
    score += risk.breaking_changes * 0.5
    if review_status.approved_reviews > 0 and not staleness.has_conflicts:
        score += 10
    if staleness.days_since_update > 30:
        score -= 5
    if staleness.days_since_update > 60:
        score -= 10
    if staleness.has_conflicts:
        score -= 15
    if review_status.changes_requested > 0:
        score -= 8
    if review_status.approved_reviews > 0 and staleness.is_mergeable:
        score += 20
    return max(0, min(100, int(score)))


# LLM Analysis Engine
def _cursor_llm_call_stub(prompt: str, pr_number: int) -> str:
    """
    This allows us to have a human-in-the-loop interface.
    The prompt is printed to the console, so Cursor can interact with it
    """
    print("\n" + "=" * 80, file=sys.stderr)
    print(f"CURSOR PROMPT FOR PR {pr_number}:", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("+" * 80, file=sys.stderr)
    return """
    **SUMMARY:**
    [PLACEHOLDER: Run the prompt above in Cursor to get this summary.]

    ###

    **Risk Assessment:**
    -   **Breaking Changes:** [PLACEHOLDER]
    -   **Security:** [PLACEHOLDER]
    -   **Urgency:** [PLACEHOLDER]
    """


def run_llm_analysis(pr_number: int, pr_title: str, pr_body: str | None, diff: str) -> dict[str, str]:
    """Here we are running an analysis with the LLM"""
    body = pr_body or "No description provided."

    # Truncate diff to avoid huge token counts
    max_diff_len = 15000  # ~4k tokens, adjustable
    if len(diff) > max_diff_len:
        diff = diff[:max_diff_len] + "\n\n... (diff truncated) ..."

    prompt = f"""
    You are a Senior Staff Software Engineer reviewing a pull request for 'lightning-thunder', a machine learning compiler.
    Analyze the following PR details and code diff.

    PR Title: {pr_title}
    PR Body:
    ---
    {body}
    ---

    Code Diff:
    ---
    {diff}
    ---

    Provide two sections in your response, separated by '###':

    **Summary:**
    [Provide a concise summary of *what* this PR does and *why*.]

    ###

    **Risk Assessment:**
    [Provide a qualitative analysis of potential risks.]
    -   **Breaking Changes:** [How likely is this to break existing user workflows? What's the reasoning?]
    -   **Security:** [Does this introduce any potential security vulnerabilities (e.g., handling untrusted inputs, credentials)?]
    -   **Urgency:** [Does this seem to fix a critical bug or blocker? Is it low priority?]
    """

    try:
        # This function now PRINTS the prompt and returns a STUB
        response_text = _cursor_llm_call_stub(prompt, pr_number)

        summary = "Could not parse LLM summary."
        risk = "Could not parse LLM risk assessment."

        if "###" in response_text:
            parts = response_text.split("###", 1)
            summary = parts[0].replace("**Summary:**", "").strip()
            risk = parts[1].replace("**Risk Assessment:**", "").strip()
        else:
            risk = response_text  # Fallback

        return {"summary": summary, "risk_assessment": risk}

    except Exception as e:
        print(f"Error in stubbed LLM analysis: {e}", file=sys.stderr)
        return {"summary": f"LLM Analysis failed: {e}", "risk_assessment": f"LLM Analysis failed: {e}"}


# Now merge the two analyses
def analyze_pr(pr_number: int) -> PRAnalysis:
    """Analyze a PR and return a PRAnalysis object"""
    print(f"Analyzing PR #{pr_number}...", file=sys.stderr)

    # 1. Fetch all GitHub data
    pr = get_pr_data(pr_number)
    reviews = get_pr_reviews(pr_number)
    comments = get_pr_comments(pr_number)
    files = get_pr_files(pr_number)
    diff = get_pr_diff(pr_number)

    # 2. Calculate staleness
    days_open = calculate_days_diff(pr["created_at"])
    days_since_update = calculate_days_diff(pr["updated_at"])
    has_conflicts = pr["mergeable"] is False

    commits_behind = None
    try:
        comparison = compare_branches(pr["head"]["sha"], pr["base"]["ref"])
        commits_behind = comparison["ahead_by"]
    except Exception as e:
        print(f"Warning: Could not get commits behind for PR #{pr_number}: {e}", file=sys.stderr)

    staleness = StalenessInfo(
        days_open=days_open,
        days_since_update=days_since_update,
        is_mergeable=pr["mergeable"],
        has_conflicts=has_conflicts,
        commits_behind_base=commits_behind,
    )

    # 3. Calculate review status
    approved_reviews = sum(1 for r in reviews if r["state"] == "APPROVED")
    changes_requested = sum(1 for r in reviews if r["state"] == "CHANGES_REQUESTED")
    pending_reviews = sum(1 for r in reviews if r["state"] == "PENDING")

    review_status = ReviewStatus(
        total_reviews=len(reviews),
        approved_reviews=approved_reviews,
        changes_requested=changes_requested,
        pending_reviews=pending_reviews,
    )

    # 4. Run Heuristic Analysis
    heuristic_risk_score = assess_risk(pr, files, comments, reviews, staleness)
    heuristic_summary = generate_summary_heuristic(pr, files, comments, reviews)

    # 5. Run LLM Analysis (Stub)
    # This will print the prompt for this PR to stderr
    llm_results = run_llm_analysis(pr["number"], pr["title"], pr.get("body"), diff)

    # 6. Calculate Final Priority (based on heuristic risk)
    priority_score = calculate_priority(pr, heuristic_risk_score, staleness, review_status)

    # 7. Combine all analysis
    return PRAnalysis(
        number=pr["number"],
        title=pr["title"],
        author=pr["user"]["login"],
        created_at=pr["created_at"],
        updated_at=pr["updated_at"],
        url=pr["html_url"],
        labels=[label["name"] for label in pr["labels"]],
        summary=heuristic_summary,
        risk_score=heuristic_risk_score,
        priority_score=priority_score,
        llm_summary=llm_results["summary"],
        llm_risk_assessment=llm_results["risk_assessment"],
        staleness=staleness,
        review_status=review_status,
        files_changed=len(files),
        additions=pr["additions"],
        deletions=pr["deletions"],
    )


# FULL MCP!
@mcp.tool()
def list_open_prs(labels: list[str] | None = None, limit: int = 50) -> str:
    """
    List all open PRs with basic information. (Heuristic only)

    Args:
        labels: Optional list of label names to filter by
        limit: Maximum number of PRs to return (default: 50)

    Returns:
        JSON string with PR list
    """
    print(f"Fetching open PRs (limit: {limit})...", file=sys.stderr)

    prs = get_open_prs(sort="created", direction="desc")

    if labels:
        labels_lower = [l.lower() for l in labels]
        filtered_prs = [pr for pr in prs if any(label["name"].lower() in labels_lower for label in pr["labels"])]
    else:
        filtered_prs = prs

    limited_prs = filtered_prs[:limit]

    result = {
        "total": len(filtered_prs),
        "showing": len(limited_prs),
        "prs": [
            {
                "number": pr["number"],
                "title": pr["title"],
                "author": pr["user"]["login"],
                "created_at": pr["created_at"],
                "updated_at": pr["updated_at"],
                "labels": [label["name"] for label in pr["labels"]],
                "url": pr["html_url"],
            }
            for pr in limited_prs
        ],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def analyze_single_pr(pr_number: int) -> str:
    """
    Get detailed heuristic analysis for a PR AND print its LLM prompt to the console.

    Args:
        pr_number: The PR number to analyze

    Returns:
        JSON string with complete PR analysis (with stubbed LLM fields)
    """
    analysis = analyze_pr(pr_number)
    return json.dumps(dataclass_to_dict(analysis), indent=2)


@mcp.tool()
def prioritize_prs(min_priority: int = 0, labels: list[str] | None = None) -> str:
    """
    Get a prioritized list of all open PRs based on *heuristic* scores.
    This is a "cheap" and "fast" analysis.

    Args:
        min_priority: Minimum priority score to include (0-100, default: 0)
        labels: Optional list of labels to filter by

    Returns:
        JSON string with prioritized PR list
    """
    print("Fetching open PRs for HEURISTIC prioritization...", file=sys.stderr)

    prs = get_open_prs(sort="created", direction="desc")

    if labels:
        labels_lower = [l.lower() for l in labels]
        prs = [pr for pr in prs if any(label["name"].lower() in labels_lower for label in pr["labels"])]

    print(f"Analyzing {len(prs)} PRs...", file=sys.stderr)

    analyses = []
    for pr_summary in prs:
        try:
            # We call analyze_pr, which will print N prompts to the console.
            # This is noisy but necessary for the function to work.
            analysis = analyze_pr(pr_summary["number"])
            if analysis.priority_score >= min_priority:
                analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing PR #{pr_summary['number']}: {e}", file=sys.stderr)

    analyses.sort(key=lambda x: x.priority_score, reverse=True)

    result = {
        "total_analyzed": len(prs),
        "meeting_criteria": len(analyses),
        "prs": [dataclass_to_dict(a) for a in analyses],
    }

    return json.dumps(result, indent=2)


# LLM prompts
@mcp.tool()
def generate_llm_priority_prompt(pr_numbers: list[int]) -> str:
    """
    Analyzes a specific list of PRs and generates a single "master prompt"
    for you to paste into Cursor. This prompt will ask the LLM to provide
    a final prioritization after you paste in the individual analyses.

    Args:
        pr_numbers: A list of PR numbers to analyze and prioritize.

    Returns:
        JSON string containing the master prompt.
    """
    print(f"Generating master prompt for PRs: {pr_numbers}...", file=sys.stderr)

    analyses = []
    print("\n--- Running individual analyses (Prompts will appear below) ---", file=sys.stderr)
    for num in pr_numbers:
        try:
            # This will print the *individual* analysis prompt for PR #{num}
            # The user should run those prompts in Cursor *first*.
            analysis = analyze_pr(num)
            analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing PR #{num}: {e}", file=sys.stderr)

    print("\n--- All individual analyses complete ---", file=sys.stderr)

    # Now, build the master prompt
    master_prompt = (
        "You are a Staff Software Engineer for the 'lightning-thunder' ML compiler.\n"
        "Your goal is to review the following set of Pull Requests and provide a final, prioritized review order.\n\n"
        "I have already run individual analyses for each PR. I will provide my 'Heuristic Analysis' and a 'Qualitative LLM Analysis' for each.\n"
        "Please use all this information to answer the final question.\n\n"
    )

    prompt_body = ""
    for a in analyses:
        prompt_body += f"""
        ---
        ### PR #{a.number}: {a.title}

        **Heuristic Analysis:**
        - Priority Score: {a.priority_score}/100
        - Risk (Overall): {a.risk_score.overall}/10 (Breaking: {a.risk_score.breaking_changes}, Security: {a.risk_score.security})
        - Urgency (if not merged): {a.risk_score.urgency_if_not_merged}/10
        - Staleness: {a.staleness.days_since_update} days since update. Conflicts: {a.staleness.has_conflicts}.

        **Qualitative LLM Analysis (Paste from Cursor):**
        [PASTE THE 'Summary' AND 'Risk Assessment' YOU GENERATED FOR PR #{a.number} HERE]

        """

    final_question = (
        "---\n\n"
        "**Final Task:**\n"
        "Based on *all* the information above, please provide:\n"
        "1.  **Prioritized List:** The order in which I should review these PRs, from most urgent/important to least.\n"
        "2.  **Brief Justification:** A one-sentence reason for each PR's position in the list.\n"
        "3.  **Overall Triage:** Are any of these safe to merge immediately? Are any immediate blockers?"
    )

    full_prompt = master_prompt + prompt_body + final_question

    # Return this as JSON so the MCP client prints it cleanly
    return json.dumps({"master_prompt_for_cursor": full_prompt}, indent=2)


@mcp.tool()
def llm_batch_analysis(
    min_priority: int = 0, labels: list[str] | None = None, limit: int = 20, include_diff: bool = False
) -> str:
    """
    Analyze ALL open PRs with heuristics, then generate a single
    comprehensive prompt for the LLM to provide its own scoring and prioritization.

    Args:
        min_priority: Only include PRs with heuristic priority >= this (0-100, default: 0)
        labels: Optional list of labels to filter by (e.g., ['bug', 'enhancement'])
        limit: Maximum number of PRs to analyze (default: 20, max recommended: 50)
        include_diff: Include code diffs in prompt (WARNING: uses many tokens, default: False)

    Returns:
        JSON string containing the comprehensive LLM analysis prompt
    """
    print("Starting LLM batch analysis")
    print(f"Input args:\n limit: {limit}, min_priority: {min_priority}), include_diff {include_diff}", file=sys.stderr)
    # Get PRs
    prs = get_open_prs(state="open", sort="created", direction="desc")
    # Filter if we have labels
    if labels:
        labels_lower = [l.lower() for l in labels]
        prs = [pr for pr in prs if any(label["name"].lower() in labels_lower for label in pr.get("labels", []))]
    print(f"Fetched {len(prs)} PRs, running heuristic analysis...", file=sys.stderr)
    # At first run a heuristic analysis
    analyses = []
    for pr_summary in prs[:limit]:
        try:
            pr = get_pr_data(pr_summary["number"])
            reviews = get_pr_reviews(pr_summary["number"])
            comments = get_pr_comments(pr_summary["number"])
            files = get_pr_files(pr_summary["number"])

            days_open = calculate_days_diff(pr["created_at"])
            days_since_update = calculate_days_diff(pr["updated_at"])
            has_conflicts = pr.get("mergeable") is False

            commits_behind = None
            try:
                comparison = compare_branches(pr["head"]["sha"], pr["base"]["ref"])
                commits_behind = comparison.get("ahead_by")
            except Exception:
                pass
            # Compute staleness
            staleness = StalenessInfo(
                days_open=days_open,
                days_since_update=days_since_update,
                is_mergeable=pr.get("mergeable"),
                has_conflicts=has_conflicts,
                commits_behind_base=commits_behind,
            )
            # Calculate review status
            approved_reviews = sum(1 for r in reviews if r["state"] == "APPROVED")
            changes_requested = sum(1 for r in reviews if r["state"] == "CHANGES_REQUESTED")
            pending_reviews = sum(1 for r in reviews if r["state"] == "PENDING")

            review_status = ReviewStatus(
                total_reviews=len(reviews),
                approved_reviews=approved_reviews,
                changes_requested=changes_requested,
                pending_reviews=pending_reviews,
            )

            # Run heuristic analysis
            heuristic_risk = assess_risk(pr, files, comments, reviews, staleness)
            priority_score = calculate_priority(pr, heuristic_risk, staleness, review_status)

            # Only include if meets priority threshold
            if priority_score >= min_priority:
                analysis_data = {
                    "number": pr["number"],
                    "title": pr["title"],
                    "author": pr["user"]["login"],
                    "url": pr["html_url"],
                    "created_at": pr["created_at"],
                    "updated_at": pr["updated_at"],
                    "labels": [label["name"] for label in pr.get("labels", [])],
                    "body_summary": (pr.get("body") or "No description")[:300].strip() + "...",
                    "files_changed": len(files),
                    "additions": pr["additions"],
                    "deletions": pr["deletions"],
                    "heuristic_priority": priority_score,
                    "risk": {
                        "overall": heuristic_risk.overall,
                        "breaking_changes": heuristic_risk.breaking_changes,
                        "security": heuristic_risk.security,
                        "urgency": heuristic_risk.urgency_if_not_merged,
                        "reasoning": {
                            "breaking": heuristic_risk.reasoning.breaking_changes,
                            "security": heuristic_risk.reasoning.security,
                            "urgency": heuristic_risk.reasoning.urgency,
                        },
                    },
                    "staleness": {
                        "days_open": days_open,
                        "days_since_update": days_since_update,
                        "has_conflicts": has_conflicts,
                        "is_mergeable": staleness.is_mergeable,
                    },
                    "review_status": {
                        "approved": approved_reviews,
                        "changes_requested": changes_requested,
                        "total_reviews": len(reviews),
                    },
                    "activity": {
                        "total_comments": len(comments),
                        "recent_activity": days_since_update < 7,
                    },
                }

                # Optionally include diff (WARNING: token-heavy)
                if include_diff:
                    try:
                        diff = get_pr_diff(pr["number"])
                        # Truncate to reasonable size
                        analysis_data["diff_preview"] = diff[:5000] + (
                            "\n\n... (truncated)" if len(diff) > 5000 else ""
                        )
                    except Exception as e:
                        print(f"Warning: Could not fetch diff for PR #{pr['number']}: {e}", file=sys.stderr)

                analyses.append(analysis_data)

        except Exception as e:
            print(f"Error analyzing PR #{pr_summary['number']}: {e}", file=sys.stderr)

    print(f"Completed heuristic analysis of {len(analyses)} PRs", file=sys.stderr)

    # Build the comprehensive LLM prompt
    prompt = _build_batch_analysis_prompt(analyses, include_diff)

    # Print prompt to stderr so user can see it
    print("\n" + "=" * 80, file=sys.stderr)
    print("🤖 LLM BATCH ANALYSIS PROMPT", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    return json.dumps(
        {
            "total_prs_analyzed": len(analyses),
            "llm_analysis_prompt": prompt,
            "heuristic_data": analyses,  # Include raw data for reference
        },
        indent=2,
    )


def _build_batch_analysis_prompt(analyses: list[dict[str, Any]], include_diff: bool) -> str:
    """Build a comprehensive prompt for LLM batch analysis."""

    analyses_sorted = sorted(analyses, key=lambda x: x["heuristic_priority"], reverse=True)

    prompt = f"""You are a Senior Staff Software Engineer for the Lightning-Thunder ML compiler project.

I need you to review and prioritize {len(analyses)} open Pull Requests. I've already performed heuristic analysis on each PR, but I need your expert judgment to provide a final assessment.

For each PR, I'm providing:
- Basic metadata (title, author, dates, labels)
- Heuristic scores (priority, risk dimensions)
- Activity metrics (reviews, comments, staleness)
- Brief description
{"- Code diff preview" if include_diff else ""}

---
## PULL REQUESTS TO ANALYZE
---

"""
    # append all the needed PRs we want to analyse
    for i, pr in enumerate(analyses_sorted, 1):
        prompt += f"""
        ### PR #{pr["number"]}: {pr["title"]}
        **Author:** @{pr["author"]}
        **URL:** {pr["url"]}
        **Labels:** {", ".join(pr["labels"]) if pr["labels"] else "None"}

        **Description:**
        {pr["body_summary"]}

        **Metrics:**
        - Created: {pr["created_at"][:10]} ({pr["staleness"]["days_open"]} days ago)
        - Last Updated: {pr["updated_at"][:10]} ({pr["staleness"]["days_since_update"]} days ago)
        - Changes: {pr["files_changed"]} files (+{pr["additions"]}/-{pr["deletions"]} lines)
        - Reviews: {pr["review_status"]["approved"]} approved, {pr["review_status"]["changes_requested"]} changes requested
        - Comments: {pr["activity"]["total_comments"]}
        - Merge Status: {"✅ No conflicts" if pr["staleness"]["is_mergeable"] else "⚠️ Has conflicts" if pr["staleness"]["has_conflicts"] else "❓ Unknown"}

        **Heuristic Analysis:**
        - Priority Score: {pr["heuristic_priority"]}/100
        - Risk Scores:
        - Overall: {pr["risk"]["overall"]}/10
        - Breaking Changes: {pr["risk"]["breaking_changes"]}/10 - {pr["risk"]["reasoning"]["breaking"]}
        - Security: {pr["risk"]["security"]}/10 - {pr["risk"]["reasoning"]["security"]}
        - Urgency: {pr["risk"]["urgency"]}/10 - {pr["risk"]["reasoning"]["urgency"]}

"""
        if include_diff and "diff_preview" in pr:
            prompt += f"""**Code Diff Preview:**
        ```diff
        {pr["diff_preview"]}
        ```
        """

    prompt += f"""
** YOUR TASK **
Please provide a comprehensive analysis with the following:

- 1. LLM priority scores (0-100 for each PR): Assign your own priority score to each PR based on:
    - Technical complexity and risk
    - Business impact and urgency
    - Code quality and readiness
    - Strategic importance to the project
    Format:
    ```
    PR #{pr["number"]}: {pr["title"]}
    Priority Score: {pr["heuristic_priority"]}/100
    Max two sentences for the reasoning to explain the choices.
    ```
    """
    prompt += """
- 2. Prioritized review order: list PRs in the order they should be reviewed, grouped by urgency:
    - 🔥 CRITICAL (Review Today):
        ...
    - 🚨 HIGH (Review This Week):
        ...
    - ⚠️ MEDIUM (Review When Possible):
        ...
    - 📝 LOW (Deprioritize):
        ...
    """
    prompt += """
- 3. Key recommendations:
        - Which PRs are safe to merge immediately?
        - Which PRs need changes before merging?
        - Which PRs are blockers for other work?
        - Which PRs should be closed/rejected and why?
        - Any PRs that need urgent attention despite low heuristic scores?
- 4. Overall assessment: A brief (2-3 paragraphs) summary of:
    - The overall health of the PR queue
    - Major themes or patterns you noticed
    - Top 3 priorities for maintainers

Please, be specific, actionable, and technical in your analysis. Consider the context of an ML compiler project
where correctness and performance are critical."""

    return prompt


# Heuristic Analysis Tools
@mcp.tool()
def check_stale_prs(days_threshold: int = 30) -> str:
    """
    Find PRs that haven't been updated recently or have merge conflicts.

    Args:
        days_threshold: Consider PRs stale if not updated in this many days (default: 30)

    Returns:
        JSON string with stale PR list
    """
    print(f"Checking for stale PRs (threshold: {days_threshold} days)...", file=sys.stderr)

    prs = get_open_prs(sort="created", direction="desc")

    print(f"Analyzing {len(prs)} PRs...", file=sys.stderr)

    analyses = []
    for pr_summary in prs:
        try:
            analysis = analyze_pr(pr_summary["number"])
            if analysis.staleness.days_since_update >= days_threshold or analysis.staleness.has_conflicts:
                analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing PR #{pr_summary['number']}: {e}", file=sys.stderr)

    analyses.sort(key=lambda x: x.staleness.days_since_update, reverse=True)

    result = {"total_stale": len(analyses), "prs": [dataclass_to_dict(a) for a in analyses]}

    return json.dumps(result, indent=2)


@mcp.tool()
def risk_report(min_risk_score: int = 5) -> str:
    """
    Generate a risk report showing high-risk PRs based on heuristic scores.

    Args:
        min_risk_score: Minimum overall heuristic risk score to include (0-10, default: 5)

    Returns:
        JSON string with risk analysis by category
    """
    print(f"Generating risk report (min score: {min_risk_score})...", file=sys.stderr)

    prs = get_open_prs(sort="created", direction="desc")

    print(f"Analyzing {len(prs)} PRs...", file=sys.stderr)

    analyses = []
    for pr_summary in prs:
        try:
            analysis = analyze_pr(pr_summary["number"])
            if analysis.risk_score.overall >= min_risk_score:
                analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing PR #{pr_summary['number']}: {e}", file=sys.stderr)

    by_breaking_changes = sorted(analyses, key=lambda x: x.risk_score.breaking_changes, reverse=True)
    by_security = sorted(analyses, key=lambda x: x.risk_score.security, reverse=True)
    by_urgency = sorted(analyses, key=lambda x: x.risk_score.urgency_if_not_merged, reverse=True)

    result = {
        "summary": {
            "total_prs_analyzed": len(prs),
            "high_risk_prs": len(analyses),
        },
        "by_breaking_changes": [dataclass_to_dict(a) for a in by_breaking_changes[:10]],
        "by_security": [dataclass_to_dict(a) for a in by_security[:10]],
        "by_urgency": [dataclass_to_dict(a) for a in by_urgency[:10]],
    }

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    try:
        print("Starting Thunder PR Inspector MCP server (Dual Analysis + Chained LLM)...")
        print("Ready for human-in-the-loop analysis with Cursor.")
        mcp.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        github_client.close()
        print("Server shut down.")
