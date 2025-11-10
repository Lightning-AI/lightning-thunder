import sys
import json
import httpx
from mcp.server.fastmcp import FastMCP
from utils.github_info import (
    get_pr_data,
    get_pr_reviews,
    get_pr_files,
    get_pr_comments,
    get_pr_diff,
    get_open_prs,
    compare_branches,
)
from utils.helper_functions import calculate_days_diff, dataclass_to_dict
from utils.constants import BASE_URL, HEADERS
from pr_scores.scores import StalenessInfo, ReviewStatus
from llm_analysis.engine import analyze_pr, build_batch_analysis_prompt
from pr_scores.heuristic import (
    assess_risk,
    assess_complexity,
    assess_impact,
    calculate_priority,
    assess_internal_review_status,
)
from plot.generate_dashboard import generate_dashboard_html, generate_dashboard_recommendations
from strategic_goals.goals_manager import get_goals_manager, StrategicGoal
from gdrive.gdrive_integration import GoogleDriveContextManager


# Create the github client (single instance, reused across all calls)
github_client = httpx.Client(base_url=BASE_URL, headers=HEADERS)

# Initialize the mcp server
mcp = FastMCP("thunder-pr-inspector")

# Initialize gdrive integration to the current MCP for gdrive
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

    prs = get_open_prs(sort="created", direction="desc", github_client=github_client)

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
    analysis = analyze_pr(pr_number, gdrive_files=gdrive_files, github_client=github_client)
    return json.dumps(dataclass_to_dict(analysis), indent=2)


@mcp.tool()
def llm_batch_analysis(
    min_priority: int = 0,
    labels: list[str] | None = None,
    limit: int = 20,
    include_diff: bool = False,
    gdrive_files: list[str] | None = None,
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
    prs = get_open_prs(state="open", sort="created", direction="desc", github_client=github_client)
    # Filter if we have labels
    if labels:
        labels_lower = [label_name.lower() for label_name in labels]
        prs = [pr for pr in prs if any(label["name"].lower() in labels_lower for label in pr.get("labels", []))]
    print(f"Fetched {len(prs)} PRs, running heuristic analysis...", file=sys.stderr)
    # At first run a heuristic analysis
    analyses = []
    for pr_summary in prs[:limit]:
        try:
            pr = get_pr_data(pr_summary["number"], github_client=github_client)
            reviews = get_pr_reviews(pr_summary["number"], github_client=github_client)
            comments = get_pr_comments(pr_summary["number"], github_client=github_client)
            files = get_pr_files(pr_summary["number"], github_client=github_client)

            days_open = calculate_days_diff(pr["created_at"])
            days_since_update = calculate_days_diff(pr["updated_at"])
            has_conflicts = pr.get("mergeable") is False

            commits_behind = None
            try:
                comparison = compare_branches(pr["head"]["sha"], pr["base"]["ref"], github_client=github_client)
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

            # Assess internal Thunder team review status
            internal_review = assess_internal_review_status(reviews, comments)

            # Assess strategic goal alignment
            goals_manager = get_goals_manager()
            goal_alignment = goals_manager.assess_pr_goal_alignment(pr)

            # Run heuristic analysis
            heuristic_risk = assess_risk(pr, files, comments, reviews, staleness)

            # Assess complexity and impact
            complexity_score, complexity_reason = assess_complexity(pr, files)
            impact_score, impact_reason = assess_impact(pr, heuristic_risk, review_status)

            # Calculate priority (with internal review status and strategic goal alignment)
            priority_score, priority_reasoning = calculate_priority(
                pr, heuristic_risk, staleness, review_status, files, internal_review, goal_alignment
            )

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
                        "category": "SIMPLE"
                        if complexity_score <= 3
                        else "MODERATE"
                        if complexity_score <= 6
                        else "COMPLEX",
                    },
                    "impact": {
                        "score": impact_score,
                        "reasoning": impact_reason,
                        "category": "LOW" if impact_score < 4 else "MEDIUM" if impact_score < 7 else "HIGH",
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
                    "internal_review": {
                        "team_approvals": internal_review.thunder_team_approvals,
                        "team_changes_requested": internal_review.thunder_team_changes_requested,
                        "team_reviewers": internal_review.thunder_team_reviewers,
                        "is_ready_for_external": internal_review.is_ready_for_external_review,
                        "status": internal_review.review_guideline_status,
                    },
                    "activity": {
                        "total_comments": len(comments),
                        "recent_activity": days_since_update < 7,
                    },
                }

                # Optionally include diff (WARNING: token-heavy)
                if include_diff:
                    try:
                        diff = get_pr_diff(pr["number"], github_client=github_client)
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
    prompt = build_batch_analysis_prompt(analyses, include_diff, gdrive_files=gdrive_files)

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
def risk_report_heuristic(min_risk_score: int = 5) -> str:
    """
    Generate a risk report showing high-risk PRs based on heuristic scores.

    Args:
        min_risk_score: Minimum overall heuristic risk score to include (0-10, default: 5)

    Returns:
        JSON string with risk analysis by category
    """
    print(f"Generating risk report (min score: {min_risk_score})...", file=sys.stderr)

    prs = get_open_prs(sort="created", direction="desc", github_client=github_client)

    print(f"Analyzing {len(prs)} PRs...", file=sys.stderr)

    analyses = []
    for pr_summary in prs:
        try:
            analysis = analyze_pr(pr_summary["number"], github_client=github_client)
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

    return json.dumps(
        {
            "instruction": "To search Google Drive, use the MaaS Google Drive MCP tool:",
            "tool_to_use": "mcp_MaaS_Google_Drive_gdrive_search",
            "parameters": {"query": query, "page_size": max_results},
            "next_steps": [
                "1. Call mcp_MaaS_Google_Drive_gdrive_search with your query",
                "2. Review the results to find relevant documents",
                "3. Use gdrive_add_context_file to add files to the PR analysis context",
                "4. Or use mcp_MaaS_Google_Drive_gdrive_get_file to fetch content directly",
            ],
            "example": {
                "query": query,
                "expected_results": "List of Google Drive files with titles, URLs, and snippets",
            },
        },
        indent=2,
    )


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
            return json.dumps(
                {"status": "error", "message": "Content cannot be empty", "file_name": file_name}, indent=2
            )

        # Add to cache
        gdrive_context.add_file_to_cache(file_name, content)

        return json.dumps(
            {
                "status": "success",
                "message": "File added to context cache",
                "file_name": file_name,
                "content_length": len(content),
                "note": "This file will be included when you use gdrive_files=['" + file_name + "']",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"Failed to add file: {str(e)}", "file_name": file_name}, indent=2
        )


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
            "preview": content[:200] + "..." if len(content) > 200 else content,
        }
        for file_name, content in gdrive_context.cache.items()
    ]

    return json.dumps(
        {
            "total_files": len(files),
            "files": files,
            "note": "These files will be included when specified in gdrive_files parameter",
        },
        indent=2,
    )


@mcp.tool()
def gdrive_clear_context_cache() -> str:
    """
    Clear all Google Drive files from the context cache.

    Returns:
        JSON string with status
    """
    count = len(gdrive_context.cache)
    gdrive_context.cache.clear()

    return json.dumps({"status": "success", "message": f"Cleared {count} files from context cache"}, indent=2)


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

    return json.dumps(
        {
            "status": "success",
            "message": message,
            "enabled": gdrive_context.gdrive_enabled,
            "cached_files": len(gdrive_context.cache),
        },
        indent=2,
    )


# DASHBOARD TOOL
@mcp.tool()
def get_review_dashboard(generate_html: bool = False) -> str:
    """
    Create a review dashboard categorizing all open PRs by their stage in the review pipeline.

    Categories:
    - Not Ready for Review: Fails "Definition of Ready" checks
    - Needs Internal Review: Ready, but has < 2 team approvals
    - Ready for External Review: Has >= 2 team approvals, no change requests
    - Blocked: Has open change requests from team members

    Args:
        generate_html: If True, generates an interactive HTML dashboard with quadrant visualization

    Returns:
        JSON string with categorized PRs and dashboard summary (includes html_path if generate_html=True)
    """
    print("Generating review dashboard...", file=sys.stderr)

    # Get all open PRs
    prs = get_open_prs(state="open", sort="created", direction="desc", github_client=github_client)
    print(f"Analyzing {len(prs)} open PRs...", file=sys.stderr)

    # Initialize categories
    not_ready = []
    needs_internal_review = []
    ready_for_external = []
    blocked = []

    for pr_summary in prs:
        try:
            pr_number = pr_summary["number"]

            # Run full analysis to get Definition of Ready and Internal Review Status
            analysis = analyze_pr(pr_number, github_client=github_client)

            # Build PR card data
            pr_card = {
                "number": analysis.number,
                "title": analysis.title,
                "author": analysis.author,
                "url": analysis.url,
                "created_at": analysis.created_at[:10],
                "updated_at": analysis.updated_at[:10],
                "labels": analysis.labels,
                "priority_score": analysis.priority_score,
                "complexity_score": analysis.complexity_score,
                "impact_score": analysis.impact_score,
                "definition_of_ready": {
                    "is_ready": analysis.definition_of_ready.is_ready,
                    "readiness_score": analysis.definition_of_ready.readiness_score,
                    "failing_checks": analysis.definition_of_ready.failing_checks,
                    "summary": analysis.definition_of_ready.readiness_summary,
                },
                "internal_review": {
                    "team_approvals": analysis.internal_review_status.thunder_team_approvals,
                    "team_changes_requested": analysis.internal_review_status.thunder_team_changes_requested,
                    "team_reviewers": analysis.internal_review_status.thunder_team_reviewers,
                    "is_ready_for_external": analysis.internal_review_status.is_ready_for_external_review,
                    "status": analysis.internal_review_status.review_guideline_status,
                },
            }

            # Categorize based on review pipeline stage
            # Priority order: Blocked > Not Ready > Ready for External > Needs Internal

            # 1. BLOCKED: Has change requests from team (highest priority to fix)
            if analysis.internal_review_status.thunder_team_changes_requested > 0:
                pr_card["block_reason"] = (
                    f"{analysis.internal_review_status.thunder_team_changes_requested} change request(s) from team"
                )
                pr_card["next_action"] = "Author needs to address change requests"
                blocked.append(pr_card)

            # 2. NOT READY FOR REVIEW: Fails Definition of Ready
            elif not analysis.definition_of_ready.is_ready:
                pr_card["not_ready_reason"] = ", ".join(analysis.definition_of_ready.failing_checks[:3])
                pr_card["next_action"] = (
                    f"Author needs to fix: {', '.join(analysis.definition_of_ready.failing_checks[:2])}"
                )
                not_ready.append(pr_card)

            # 3. READY FOR EXTERNAL REVIEW: Has >= 2 team approvals
            elif analysis.internal_review_status.is_ready_for_external_review:
                pr_card["ready_details"] = f"{analysis.internal_review_status.thunder_team_approvals} team approvals"
                pr_card["next_action"] = "Ping external maintainers (@lantiga, @t-vi, @KaelanDt)"
                ready_for_external.append(pr_card)

            # 4. NEEDS INTERNAL REVIEW: Ready but < 2 team approvals
            else:
                approvals_needed = 2 - analysis.internal_review_status.thunder_team_approvals
                pr_card["needs_approvals"] = approvals_needed
                pr_card["next_action"] = f"Needs {approvals_needed} more team approval(s)"
                needs_internal_review.append(pr_card)

        except Exception as e:
            print(f"Error analyzing PR #{pr_summary['number']}: {e}", file=sys.stderr)

    # Sort each category by priority score (descending)
    not_ready.sort(key=lambda x: x["priority_score"], reverse=True)
    needs_internal_review.sort(key=lambda x: x["priority_score"], reverse=True)
    ready_for_external.sort(key=lambda x: x["priority_score"], reverse=True)
    blocked.sort(key=lambda x: x["priority_score"], reverse=True)

    # Build summary statistics
    total_prs = len(prs)
    summary = {
        "total_open_prs": total_prs,
        "not_ready": len(not_ready),
        "needs_internal_review": len(needs_internal_review),
        "ready_for_external": len(ready_for_external),
        "blocked": len(blocked),
        "pipeline_health": {
            "ready_to_merge": len(ready_for_external),
            "in_review": len(needs_internal_review),
            "needs_attention": len(blocked) + len(not_ready),
            "blocked_percentage": int(len(blocked) / total_prs * 100) if total_prs > 0 else 0,
            "ready_percentage": int(len(ready_for_external) / total_prs * 100) if total_prs > 0 else 0,
        },
    }

    # Build dashboard
    dashboard = {
        "summary": summary,
        "categories": {
            "blocked": {
                "count": len(blocked),
                "emoji": "🔄",
                "description": "PRs with open change requests from team members",
                "priority": "CRITICAL - Author action required",
                "prs": blocked[:20],  # Limit to top 20 for display
            },
            "not_ready_for_review": {
                "count": len(not_ready),
                "emoji": "🚫",
                "description": "PRs that fail Definition of Ready checks",
                "priority": "HIGH - Author action required",
                "prs": not_ready[:20],
            },
            "ready_for_external_review": {
                "count": len(ready_for_external),
                "emoji": "✅",
                "description": "PRs with >= 2 team approvals, ready for external maintainers",
                "priority": "HIGH - Ready to ping @lantiga, @t-vi, @KaelanDt",
                "prs": ready_for_external[:20],
            },
            "needs_internal_review": {
                "count": len(needs_internal_review),
                "emoji": "⏳",
                "description": "PRs ready for review but need more team approvals",
                "priority": "MEDIUM - Team review needed",
                "prs": needs_internal_review[:20],
            },
        },
        "recommendations": generate_dashboard_recommendations(
            summary, ready_for_external, needs_internal_review, blocked, not_ready
        ),
    }

    # Print summary to stderr for console visibility
    print("\n" + "=" * 80, file=sys.stderr)
    print("📊 REVIEW DASHBOARD SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Total Open PRs: {total_prs}", file=sys.stderr)
    print(f"  🔄 Blocked: {len(blocked)} ({summary['pipeline_health']['blocked_percentage']}%)", file=sys.stderr)
    print(f"  🚫 Not Ready: {len(not_ready)}", file=sys.stderr)
    print(
        f"  ✅ Ready for External: {len(ready_for_external)} ({summary['pipeline_health']['ready_percentage']}%)",
        file=sys.stderr,
    )
    print(f"  ⏳ Needs Internal Review: {len(needs_internal_review)}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Generate HTML dashboard if requested
    if generate_html:
        try:
            html_path = generate_dashboard_html(
                blocked=blocked,
                not_ready=not_ready,
                needs_internal_review=needs_internal_review,
                ready_for_external=ready_for_external,
                summary=summary,
            )
            dashboard["html_dashboard_path"] = html_path
            print(f"✅ HTML dashboard generated: {html_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Failed to generate HTML dashboard: {e}", file=sys.stderr)
            dashboard["html_generation_error"] = str(e)

    return json.dumps(dashboard, indent=2)


@mcp.tool()
def add_strategic_goal(
    goal_id: str, title: str, priority: str, description: str, theme: str, linked_issues: list[int] | None = None
) -> str:
    """
    Add a strategic goal (e.g., Q4 priority) to track PR alignment in the final priority score.

    Args:
        goal_id: Unique identifier (e.g., "Q4-inference-opt")
        title: Goal title (e.g., "Inference Optimization")
        priority: "P0", "P1", or "P2"
        description: Detailed description
        theme: Theme category (e.g., "Performance", "Features")
        linked_issues: Optional list of GitHub issue numbers

    Returns:
        JSON confirmation

    Example:
        add_strategic_goal(
            goal_id="Q4-inference-opt",
            title="Inference Optimization",
            priority="P0",
            description="Optimize inference performance for production workloads",
            theme="Performance",
            linked_issues=[2556, 2557]
        )
    """
    goals_manager = get_goals_manager()

    goal = StrategicGoal(
        id=goal_id,
        title=title,
        priority=priority,
        description=description,
        linked_issues=linked_issues or [],
        theme=theme,
    )

    goals_manager.add_goal(goal)

    return json.dumps(
        {
            "status": "success",
            "message": f"Added strategic goal: {title}",
            "goal": {
                "id": goal_id,
                "title": title,
                "priority": priority,
                "linked_issues": len(goal.linked_issues),
                "theme": theme,
            },
        },
        indent=2,
    )


@mcp.tool()
def link_issue_to_goal(issue_number: int, goal_id: str) -> str:
    """
    Link a GitHub issue to a strategic goal.

    Args:
        issue_number: GitHub issue number
        goal_id: Strategic goal ID

    Returns:
        JSON confirmation
    """
    goals_manager = get_goals_manager()
    goals_manager.link_issue_to_goal(issue_number, goal_id)

    return json.dumps({"status": "success", "message": f"Linked issue #{issue_number} to goal '{goal_id}'"}, indent=2)


@mcp.tool()
def list_strategic_goals() -> str:
    """
    List all configured strategic goals.

    Returns:
        JSON with all goals and their linked issues
    """
    goals_manager = get_goals_manager()

    goals_list = []
    for goal_id, goal in goals_manager.goals.items():
        goals_list.append(
            {
                "id": goal.id,
                "title": goal.title,
                "priority": goal.priority,
                "theme": goal.theme,
                "description": goal.description,
                "linked_issues": goal.linked_issues,
                "issue_count": len(goal.linked_issues),
            }
        )

    # Sort by priority (P0 first)
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    goals_list.sort(key=lambda g: priority_order.get(g["priority"], 999))

    return json.dumps({"total_goals": len(goals_list), "goals": goals_list}, indent=2)


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
