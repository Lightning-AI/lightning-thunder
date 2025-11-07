import os
import sys
from typing import Any
import json
import httpx
from mcp.server.fastmcp import FastMCP
from utils.github_info import (get_pr_data, get_pr_reviews, get_pr_files, get_pr_comments, get_pr_diff, get_open_prs, compare_branches)
from utils.helper_functions import calculate_days_diff, dataclass_to_dict
from pr_scores.scores import (StalenessInfo, ReviewStatus)
from llm_analysis.engine import analyze_pr
from pr_scores.heuristic import assess_risk, assess_complexity, assess_impact, calculate_priority
from gdrive.gdrive_integration import GoogleDriveContextManager


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
# this is a gdrive integration to the current MCP for gdrive
gdrive_context = GoogleDriveContextManager()


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
        labels_lower = [label_name.lower() for label_name in labels]
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
def analyze_single_pr(pr_number: int, gdrive_files: list[str] | None = None) -> str:
    """
    Get detailed heuristic analysis for a single PR AND print its LLM prompt to the console.

    Args:
        pr_number: The PR number to analyze
        gdrive_files: Optional list of Google Drive file names or URLs to use as context
                     e.g., ["ThunderQ4Plan", "ThunderBestPractices"]
                     Leave empty to analyze without Google Drive context

    Returns:
        JSON string with complete PR analysis (with stubbed LLM fields)
        
    Examples:
        # Without context:
        analyze_single_pr(pr_number=123)
        
        # With specific Google Drive files:
        analyze_single_pr(
            pr_number=123, 
            gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"]
        )
    """
    analysis = analyze_pr(pr_number, gdrive_files=gdrive_files)
    return json.dumps(dataclass_to_dict(analysis), indent=2)


@mcp.tool()
def prioritize_heuristic_prs(min_priority: int = 0, labels: list[str] | None = None) -> str:
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
        labels_lower = [label_name.lower() for label_name in labels]
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


# @mcp.tool()
# def generate_llm_priority_prompt(pr_numbers: list[int]) -> str:
#     """
#     Analyzes a specific list of PRs and generates a single "master prompt"
#     for you to paste into Cursor. This prompt will ask the LLM to provide
#     a final prioritization after you paste in the individual analyses.

#     Args:
#         pr_numbers: A list of PR numbers to analyze and prioritize.

#     Returns:
#         JSON string containing the master prompt.
#     """
#     print(f"Generating master prompt for PRs: {pr_numbers}...", file=sys.stderr)

#     analyses = []
#     print("\n--- Running individual analyses (Prompts will appear below) ---", file=sys.stderr)
#     for num in pr_numbers:
#         try:
#             # This will print the *individual* analysis prompt for PR #{num}
#             # The user should run those prompts in Cursor *first*.
#             analysis = analyze_pr(num)
#             analyses.append(analysis)
#         except Exception as e:
#             print(f"Error analyzing PR #{num}: {e}", file=sys.stderr)

#     print("\n--- All individual analyses complete ---", file=sys.stderr)

#     # Now, build the master prompt
#     master_prompt = (
#         "You are a Staff Software Engineer for the 'lightning-thunder' ML compiler.\n"
#         "Your goal is to review the following set of Pull Requests and provide a final, prioritized review order.\n\n"
#         "I have already run individual analyses for each PR. I will provide my 'Heuristic Analysis' and a 'Qualitative LLM Analysis' for each.\n"
#         "Please use all this information to answer the final question.\n\n"
#     )

#     prompt_body = ""
#     for a in analyses:
#         prompt_body += f"""
#         ---
#         ### PR #{a.number}: {a.title}

#         **Heuristic Analysis:**
#         - Priority Score: {a.priority_score}/100
#         - Risk (Overall): {a.risk_score.overall}/10 (Breaking: {a.risk_score.breaking_changes}, Security: {a.risk_score.security})
#         - Urgency (if not merged): {a.risk_score.urgency_if_not_merged}/10
#         - Staleness: {a.staleness.days_since_update} days since update. Conflicts: {a.staleness.has_conflicts}.

#         **Qualitative LLM Analysis (Paste from Cursor):**
#         [PASTE THE 'Summary' AND 'Risk Assessment' YOU GENERATED FOR PR #{a.number} HERE]

#         """

#     final_question = (
#         "---\n\n"
#         "**Final Task:**\n"
#         "Based on *all* the information above, please provide:\n"
#         "1.  **Prioritized List:** The order in which I should review these PRs, from most urgent/important to least.\n"
#         "2.  **Brief Justification:** A one-sentence reason for each PR's position in the list.\n"
#         "3.  **Overall Triage:** Are any of these safe to merge immediately? Are any immediate blockers?"
#     )

#     full_prompt = master_prompt + prompt_body + final_question

#     # Return this as JSON so the MCP client prints it cleanly
#     return json.dumps({"master_prompt_for_cursor": full_prompt}, indent=2)


@mcp.tool()
def llm_batch_analysis(
    min_priority: int = 0, 
    labels: list[str] | None = None, 
    limit: int = 20, 
    include_diff: bool = False,
    gdrive_files: list[str] | None = None
) -> str:
    """
    Analyze ALL open PRs with heuristics, then generate a single
    comprehensive prompt for the LLM to provide its own scoring and prioritization.

    Args:
        min_priority: Only include PRs with heuristic priority >= this (0-100, default: 0)
        labels: Optional list of labels to filter by (e.g., ['bug', 'enhancement'])
        limit: Maximum number of PRs to analyze (default: 20, max recommended: 50)
        include_diff: Include code diffs in prompt (WARNING: uses many tokens, default: False)
        gdrive_files: Optional list of Google Drive file names or URLs to use as context
                     e.g., ["ThunderQ4Plan", "ThunderBestPractices"]

    Returns:
        JSON string containing the comprehensive LLM analysis prompt
        
    Examples:
        # Without Google Drive context:
        llm_batch_analysis(min_priority=30, limit=10)
        
        # With specific Google Drive files for calibration:
        llm_batch_analysis(
            min_priority=30,
            limit=10,
            gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"]
        )
    """
    print("Starting LLM batch analysis")
    print(f"Input args:\n limit: {limit}, min_priority: {min_priority}), include_diff {include_diff}", file=sys.stderr)
    # Get PRs
    prs = get_open_prs(state="open", sort="created", direction="desc")
    # Filter if we have labels
    if labels:
        labels_lower = [label_name.lower() for label_name in labels]
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
            
            # Assess complexity and impact
            complexity_score, complexity_reason = assess_complexity(pr, files)
            impact_score, impact_reason = assess_impact(pr, heuristic_risk, review_status)
            
            # Calculate priority
            priority_score, priority_reasoning = calculate_priority(pr, heuristic_risk, staleness, review_status, files)

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
                    "priority_reasoning": priority_reasoning,
                    "complexity": {
                        "score": complexity_score,
                        "reasoning": complexity_reason,
                        "category": "SIMPLE" if complexity_score <= 3 else "MODERATE" if complexity_score <= 6 else "COMPLEX"
                    },
                    "impact": {
                        "score": impact_score,
                        "reasoning": impact_reason,
                        "category": "LOW" if impact_score < 4 else "MEDIUM" if impact_score < 7 else "HIGH"
                    },
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

    # Build the comprehensive LLM prompt with optional Google Drive context
    prompt = _build_batch_analysis_prompt(analyses, include_diff, gdrive_files=gdrive_files)

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
def risk_report_heuristic(min_risk_score: int = 5) -> str:
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


# ============================================================================
# GOOGLE DRIVE MCP TOOLS
# ============================================================================

@mcp.tool()
def gdrive_search_docs(query: str, max_results: int = 5) -> str:
    """
    Search Google Drive for documents relevant to PR review.
    
    This tool allows you to search your organization's Google Drive for
    documents like coding standards, architecture docs, or other relevant
    materials that can enhance PR analysis.
    
    Args:
        query: Search query (e.g., "lightning-thunder coding standards")
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        JSON string with search results
        
    NOTE: This tool provides guidance on using the MaaS Google Drive MCP.
    To actually search, you should use the mcp_MaaS_Google_Drive_gdrive_search
    tool available in Cursor.
    """
    print(f"🔍 Searching Google Drive for: {query}", file=sys.stderr)
    
    return json.dumps({
        "instruction": "To search Google Drive, use the MaaS Google Drive MCP tool:",
        "tool_to_use": "mcp_MaaS_Google_Drive_gdrive_search",
        "parameters": {
            "query": query,
            "page_size": max_results
        },
        "next_steps": [
            "1. Call mcp_MaaS_Google_Drive_gdrive_search with your query",
            "2. Review the results to find relevant documents",
            "3. Use gdrive_add_context_file to add files to the PR analysis context",
            "4. Or use mcp_MaaS_Google_Drive_gdrive_get_file to fetch content directly"
        ],
        "example": {
            "query": query,
            "expected_results": "List of Google Drive files with titles, URLs, and snippets"
        }
    }, indent=2)


@mcp.tool()
def gdrive_add_context_file(file_name: str, content: str) -> str:
    """
    Add a Google Drive file's content to the PR analysis context cache.
    
    This caches file content so it can be used in subsequent PR analyses
    without re-fetching from Google Drive.
    
    Args:
        file_name: Name of the file (e.g., "ThunderQ4Plan")
        content: The file content (fetch via mcp_MaaS_Google_Drive_gdrive_get_file first)
        
    Returns:
        JSON string with status
        
    Usage:
        1. Search for file: mcp_MaaS_Google_Drive_gdrive_search(query="ThunderQ4Plan")
        2. Get content: content = mcp_MaaS_Google_Drive_gdrive_get_file(file_url="...")
        3. Cache it: gdrive_add_context_file(file_name="ThunderQ4Plan", content=content)
        4. Analyze PRs: analyze_single_pr(123, gdrive_files=["ThunderQ4Plan"])
    """
    print(f"📥 Adding file to cache: {file_name}", file=sys.stderr)
    
    try:
        if not content:
            return json.dumps({
                "status": "error",
                "message": "Content cannot be empty",
                "file_name": file_name
            }, indent=2)
        
        # Add to cache
        gdrive_context.add_file_to_cache(file_name, content)
        
        return json.dumps({
            "status": "success",
            "message": "File added to context cache",
            "file_name": file_name,
            "content_length": len(content),
            "note": "This file will be included when you use gdrive_files=['" + file_name + "']"
        }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to add file: {str(e)}",
            "file_name": file_name
        }, indent=2)


@mcp.tool()
def gdrive_list_context_files() -> str:
    """
    List all Google Drive files currently in the context cache.
    
    Returns:
        JSON string with list of cached files
    """
    files = [
        {
            "file_name": file_name,
            "content_length": len(content),
            "preview": content[:200] + "..." if len(content) > 200 else content
        }
        for file_name, content in gdrive_context.cache.items()
    ]
    
    return json.dumps({
        "total_files": len(files),
        "files": files,
        "note": "These files will be included when specified in gdrive_files parameter"
    }, indent=2)


@mcp.tool()
def gdrive_clear_context_cache() -> str:
    """
    Clear all Google Drive files from the context cache.
    
    Returns:
        JSON string with status
    """
    count = len(gdrive_context.cache)
    gdrive_context.cache.clear()
    
    return json.dumps({
        "status": "success",
        "message": f"Cleared {count} files from context cache"
    }, indent=2)


@mcp.tool()
def gdrive_configure(enabled: bool = True) -> str:
    """
    Enable or disable Google Drive integration.
    
    Args:
        enabled: Whether to enable Google Drive context fetching
        
    Returns:
        JSON string with status
    """
    if enabled:
        gdrive_context.gdrive_enabled = True
        message = "Google Drive integration enabled"
    else:
        gdrive_context.disable()
        message = "Google Drive integration disabled"
    
    return json.dumps({
        "status": "success",
        "message": message,
        "enabled": gdrive_context.gdrive_enabled,
        "cached_files": len(gdrive_context.cache)
    }, indent=2)


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
