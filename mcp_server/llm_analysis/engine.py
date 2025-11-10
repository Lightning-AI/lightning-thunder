import sys
import httpx
from typing import Any
from utils.github_info import (
    get_pr_data,
    get_pr_reviews,
    get_pr_comments,
    get_pr_files,
    get_pr_diff,
    compare_branches,
    get_ci_check_runs,
)
from utils.helper_functions import calculate_days_diff
from pr_scores.scores import PRAnalysis, StalenessInfo, ReviewStatus
from pr_scores.heuristic import (
    assess_risk,
    assess_complexity,
    assess_impact,
    calculate_priority,
    assess_internal_review_status,
    check_definition_of_ready,
    generate_summary_heuristic,
)
from strategic_goals.goals_manager import get_goals_manager
from gdrive.gdrive_integration import GoogleDriveContextManager
from pr_scores.scores import GoalAlignment


# Initialize Google Drive context manager
gdrive_context = GoogleDriveContextManager()


def format_analysis_for_display(analysis: PRAnalysis) -> str:
    """
    Format the PR analysis as a readable markdown string.

    Args:
        analysis: PRAnalysis object containing all the analysis data

    Returns:
        Formatted markdown string ready for display
    """
    # Status emoji
    ready_emoji = "✅" if analysis.definition_of_ready.is_ready else "⚠️"
    ready_status = "Ready for Review" if analysis.definition_of_ready.is_ready else "Not Ready for Review"

    # Priority category
    if analysis.priority_score >= 80:
        priority_category = "🔥 HIGH PRIORITY"
    elif analysis.priority_score >= 60:
        priority_category = "⚡ MEDIUM PRIORITY"
    else:
        priority_category = "📋 STANDARD PRIORITY"

    # Build the formatted output
    output = f"""# PR #{analysis.number}: {analysis.title}

**Author:** {analysis.author}
**Created:** {analysis.created_at[:10]} ({analysis.staleness.days_open} days ago)
**URL:** {analysis.url}
**Status:** {ready_emoji} {ready_status}

## Overview
{analysis.summary}

## Priority & Scores
- **Priority:** {analysis.priority_score}/100 - {priority_category}
- **Complexity:** {analysis.complexity_score}/10
- **Impact:** {analysis.impact_score}/10
- **Risk:** {analysis.risk_score.overall}/10

**Priority Reasoning:**
{analysis.priority_reasoning}

## Changes
- Files Changed: {analysis.files_changed}
- Additions: +{analysis.additions}
- Deletions: -{analysis.deletions}
- Labels: {", ".join(analysis.labels) if analysis.labels else "None"}

## Definition of Ready
**Readiness Score:** {analysis.definition_of_ready.readiness_score}/100

{analysis.definition_of_ready.readiness_summary}
"""

    # Add failing checks if any
    if analysis.definition_of_ready.failing_checks:
        output += "\n**Failing Checks:**\n"
        for check in analysis.definition_of_ready.failing_checks:
            output += f"- ❌ {check}\n"

    # Internal review status
    output += f"""
## Internal Review Status (Thunder Team)
{analysis.internal_review_status.review_guideline_status}

- Team Approvals: {analysis.internal_review_status.thunder_team_approvals}/2
- Changes Requested: {analysis.internal_review_status.thunder_team_changes_requested}
- Ready for External Review: {"Yes ✅" if analysis.internal_review_status.is_ready_for_external_review else "No ⏳"}
"""

    if analysis.internal_review_status.thunder_team_reviewers:
        output += f"- Reviewers: {', '.join(analysis.internal_review_status.thunder_team_reviewers)}\n"

    # External review status
    output += f"""
## External Review Status
- Total Reviews: {analysis.review_status.total_reviews}
- Approved: {analysis.review_status.approved_reviews}
- Changes Requested: {analysis.review_status.changes_requested}
- Pending: {analysis.review_status.pending_reviews}
"""

    # Strategic goal alignment
    if analysis.goal_alignment:
        output += f"""
## Strategic Goal Alignment
**Aligned:** {"Yes ✅" if analysis.goal_alignment.is_aligned else "No ⚠️"}
**Alignment Score:** {analysis.goal_alignment.alignment_score}/100

{analysis.goal_alignment.alignment_reasoning}
"""
        if analysis.goal_alignment.highest_priority:
            output += f"\n**Highest Priority Goal:** {analysis.goal_alignment.highest_priority}\n"

        if analysis.goal_alignment.closed_issues:
            output += (
                "\n**Closed Issues:** "
                + ", ".join(f"#{issue}" for issue in analysis.goal_alignment.closed_issues)
                + "\n"
            )

    # Risk assessment
    output += f"""
## Risk Assessment
**Overall Risk:** {analysis.risk_score.overall}/10

- **Breaking Changes:** {analysis.risk_score.breaking_changes}/10
  - {analysis.risk_score.reasoning.breaking_changes}
- **Security:** {analysis.risk_score.security}/10
  - {analysis.risk_score.reasoning.security}
- **Urgency:** {analysis.risk_score.urgency_if_not_merged}/10
  - {analysis.risk_score.reasoning.urgency}
"""

    # Staleness info
    if analysis.staleness.has_conflicts:
        output += "\n⚠️ **WARNING:** This PR has merge conflicts!\n"

    if analysis.staleness.commits_behind_base and analysis.staleness.commits_behind_base > 10:
        output += f"\n⚠️ **WARNING:** This PR is {analysis.staleness.commits_behind_base} commits behind base branch\n"

    # LLM Analysis (if not placeholder)
    if "[PLACEHOLDER" not in analysis.llm_summary:
        output += f"""
## LLM Analysis

### Summary
{analysis.llm_summary}

### Risk Assessment
{analysis.llm_risk_assessment}
"""

    output += "\n---\n*Generated by Thunder PR Analysis MCP*\n"

    return output


def _cursor_llm_call_stub(prompt: str, pr_number: int) -> tuple[str, str]:
    """
    This allows us to have a human-in-the-loop interface.
    The prompt is printed to the console, so Cursor can interact with it

    Args:
        prompt: The prompt to print
        pr_number: The PR number

    Returns:
        Tuple of (placeholder_response, actual_prompt)
    """
    print("\n" + "=" * 80, file=sys.stderr)
    print(f"CURSOR PROMPT FOR PR {pr_number}:", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("+" * 80, file=sys.stderr)

    placeholder = """
    **SUMMARY:**
    [PLACEHOLDER: Run the prompt above in Cursor to get this summary.]

    ###

    **Risk Assessment:**
    -   **Breaking Changes:** [PLACEHOLDER]
    -   **Security:** [PLACEHOLDER]
    -   **Urgency:** [PLACEHOLDER]
    """
    return placeholder, prompt


def run_llm_analysis(
    pr_number: int,
    pr_title: str,
    pr_body: str | None,
    diff: str,
    gdrive_files: list[str] | None = None,
    heuristic_scores: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Run LLM analysis on a single PR with optional Google Drive context and heuristic scores.

    Args:
        pr_number: PR number
        pr_title: PR title
        pr_body: PR description
        diff: PR diff
        gdrive_files: List of Google Drive file names or URLs to use as context
                     e.g., ["ThunderQ4Plan", "ThunderBestPractices"]
        heuristic_scores: Dictionary with complexity, impact, priority scores and reasoning

    Returns:
        Dictionary with 'summary' and 'risk_assessment' keys
    """
    body = pr_body or "No description provided."

    # Truncate diff to avoid huge token counts
    max_diff_len = 15000  # ~4k tokens, adjustable
    if len(diff) > max_diff_len:
        diff = diff[:max_diff_len] + "\n\n... (diff truncated) ..."

    # Build additional context from specified Google Drive files
    additional_context = ""
    if gdrive_files:
        try:
            additional_context = gdrive_context.build_context_from_files(gdrive_files)
            if additional_context:
                print(
                    f"✓ Added Google Drive context to PR #{pr_number} analysis ({len(gdrive_files)} files)",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"Warning: Failed to fetch Google Drive context: {e}", file=sys.stderr)

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
    """

    # Add heuristic analysis context if provided
    if heuristic_scores:
        complexity = heuristic_scores.get("complexity", 0)
        impact = heuristic_scores.get("impact", 0)
        priority = heuristic_scores.get("priority", 0)
        priority_reasoning = heuristic_scores.get("priority_reasoning", "")

        # Determine complexity category
        if complexity <= 3:
            complexity_cat = "SIMPLE"
            review_guidance = "This is a straightforward change. Focus on correctness and alignment with standards."
        elif complexity <= 6:
            complexity_cat = "MODERATE"
            review_guidance = "This is a moderately complex change. Check for edge cases and integration issues."
        else:
            complexity_cat = "COMPLEX"
            review_guidance = (
                "This is a complex change. Pay special attention to architecture, testing, and potential side effects."
            )

        # Determine impact category
        if impact >= 7:
            impact_cat = "HIGH IMPACT"
        elif impact >= 4:
            impact_cat = "MEDIUM IMPACT"
        else:
            impact_cat = "LOW IMPACT"

        prompt += f"""

    ## HEURISTIC ANALYSIS

    Our automated system has analyzed this PR:

    **Complexity Score:** {complexity}/10 ({complexity_cat})
    **Impact Score:** {impact}/10 ({impact_cat})
    **Priority Score:** {priority}/100

    **Priority Reasoning:**
    {priority_reasoning}

    **Review Guidance:** {review_guidance}

    ---
    """

    # Add reference documentation if provided
    if additional_context:
        prompt += f"""

    {additional_context}

    Please use the above documentation to calibrate your assessment of this PR.
    Your analysis should align with the goals, standards, and priorities outlined in these documents.
    ---
    """

    # Adjust prompt based on complexity
    if heuristic_scores and heuristic_scores.get("complexity", 0) >= 7:
        # Complex PR - ask for detailed debugging guidance
        prompt += """

    Provide THREE sections in your response, separated by '###':

    **Summary:**
    [Provide a concise summary of *what* this PR does and *why*.]

    ###

    **Risk Assessment:**
    [Provide a qualitative analysis of potential risks.]
    -   **Breaking Changes:** [How likely is this to break existing user workflows? What's the reasoning?]
    -   **Security:** [Does this introduce any potential security vulnerabilities (e.g., handling untrusted inputs, credentials)?]
    -   **Urgency:** [Does this seem to fix a critical bug or blocker? Is it low priority?]

    ###

    **Review Checklist & Debugging Guide:**
    [Since this is a COMPLEX PR, provide specific guidance:]
    -   **Key Areas to Review:** [Which parts of the code are most critical? What should reviewers focus on?]
    -   **Potential Issues:** [What could go wrong? What edge cases should be tested?]
    -   **Testing Strategy:** [What testing approach would you recommend? What scenarios must be covered?]
    -   **Architecture Impact:** [How does this affect the overall system architecture? Any concerns?]
    -   **Debug Checklist:** [If this causes issues in production, what should be checked first?]
    """
    else:
        # Simple/Moderate PR - standard analysis
        prompt += """

    Provide two sections in your response, separated by '\n':

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
        # This function now PRINTS the prompt and returns a STUB + the actual prompt
        response_text, actual_prompt = _cursor_llm_call_stub(prompt, pr_number)

        summary = "Could not parse LLM summary."
        risk = "Could not parse LLM risk assessment."

        if "###" in response_text:
            parts = response_text.split("###", 1)
            summary = parts[0].replace("**Summary:**", "").strip()
            risk = parts[1].replace("**Risk Assessment:**", "").strip()
        else:
            risk = response_text  # Fallback

        return {
            "summary": summary,
            "risk_assessment": risk,
            "llm_prompt": actual_prompt,  # Include the prompt for easy access
        }

    except Exception as e:
        print(f"Error in stubbed LLM analysis: {e}", file=sys.stderr)
        return {
            "summary": f"LLM Analysis failed: {e}",
            "risk_assessment": f"LLM Analysis failed: {e}",
            "llm_prompt": prompt,  # Still include the prompt even on error
        }


def build_batch_analysis_prompt(
    analyses: list[dict[str, Any]], include_diff: bool, gdrive_files: list[str] | None = None
) -> str:
    """
    Build a comprehensive prompt for LLM batch analysis with optional Google Drive context.

    Args:
        analyses: List of PR analysis data
        include_diff: Whether to include code diffs
        gdrive_files: Optional list of Google Drive file names or URLs to use as context

    Returns:
        Formatted prompt string
    """

    analyses_sorted = sorted(analyses, key=lambda x: x["heuristic_priority"], reverse=True)

    # Fetch specified files from Google Drive
    project_context = ""
    if gdrive_files:
        try:
            project_context = gdrive_context.build_context_from_files(gdrive_files)
            if project_context:
                project_context = f"\n{project_context}\n\n"
                print(f"✓ Added {len(gdrive_files)} Google Drive files to batch analysis", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to fetch Google Drive context for batch analysis: {e}", file=sys.stderr)

    prompt = f"""You are a Senior Staff Software Engineer for the Lightning-Thunder ML compiler project.

I need you to review and prioritize {len(analyses)} open Pull Requests. I've already performed heuristic analysis on each PR, but I need your expert judgment to provide a final assessment.

For each PR, I'm providing:
- Basic metadata (title, author, dates, labels)
- Heuristic scores (priority, risk dimensions)
- Activity metrics (reviews, comments, staleness)
- Brief description
{"- Code diff preview" if include_diff else ""}
"""

    # Add reference documentation if provided
    if project_context:
        prompt += f"""
---
## REFERENCE DOCUMENTATION
---

{project_context}

Please use the above documentation to calibrate your analysis and prioritization of the PRs below.
Your assessment should align with the goals, standards, and priorities outlined in these documents.
"""

    prompt += """
---
## PULL REQUESTS TO ANALYZE
---

"""
    # append all the needed PRs we want to analyse
    for i, pr in enumerate(analyses_sorted, 1):
        # Get complexity/impact categories for visual indicators
        complexity_emoji = "🟢" if pr["complexity"]["score"] <= 3 else "🟡" if pr["complexity"]["score"] <= 6 else "🔴"
        impact_emoji = "🔵" if pr["impact"]["score"] < 4 else "🟠" if pr["impact"]["score"] < 7 else "🔴"

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

        **Internal Review Status (Thunder Team):**
        - {pr["internal_review"]["status"]}
        - Team Approvals: {pr["internal_review"]["team_approvals"]}/2 required
        - Team Reviewers: {", ".join(["@" + r for r in pr["internal_review"]["team_reviewers"]]) if pr["internal_review"]["team_reviewers"] else "None yet"}
        - Ready for External Review: {"✅ Yes" if pr["internal_review"]["is_ready_for_external"] else "⏳ No"}

        **Heuristic Analysis:**
        - Priority Score: {pr["heuristic_priority"]}/100
        - {complexity_emoji} Complexity: {pr["complexity"]["score"]}/10 ({pr["complexity"]["category"]}) - {pr["complexity"]["reasoning"]}
        - {impact_emoji} Impact: {pr["impact"]["score"]}/10 ({pr["impact"]["category"]}) - {pr["impact"]["reasoning"]}
        - Priority Reasoning:
          {pr["priority_reasoning"]}

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

    prompt += """
** YOUR TASK **
You've been given heuristic scores (complexity, impact, priority) for each PR.
Use these as a STARTING POINT, but apply your expert judgment to provide final recommendations.

Please provide a comprehensive analysis with the following:

**1. Enhanced Priority Assessment (for each PR):**
For each PR, considering its complexity and impact:

   For SIMPLE PRs (🟢 Complexity ≤ 3):
   - Quick review checklist: What to verify?
   - Estimated review time
   - Can it be merged quickly?

   For MODERATE PRs (🟡 Complexity 4-6):
   - Key areas to focus on
   - Integration concerns
   - Testing recommendations

   For COMPLEX PRs (🔴 Complexity ≥ 7):
   - Detailed review strategy
   - What could go wrong? (potential issues)
   - Debug checklist if this causes problems
   - Recommendation: Should this be broken into smaller PRs?

**2. Prioritized Review Order:**
Group PRs by urgency, considering complexity × impact:
   - 🔥 CRITICAL (Review Today): Simple + High Impact, or Critical issues
   - ⚡ QUICK WINS (Review This Week): Simple + Any Impact (easy to clear)
   - 🎯 IMPORTANT (Schedule Deep Review): Complex + High Impact (needs time)
   - 📝 LOW (Deprioritize): Complex + Low Impact

**3. Complexity-Specific Guidance:**
   For each COMPLEX PR (🔴):
   - Break down: What makes it complex?
   - Risk areas: Where should reviewers be extra careful?
   - Testing strategy: What must be tested?
   - Simplification opportunities: Can complexity be reduced?

**4. Key Recommendations:**
   - Which PRs are safe to merge immediately? (Simple + Approved)
   - Which PRs need changes before merging?
   - Which PRs are blockers for other work?
   - Which PRs should be closed/rejected and why?
   - Any PRs that need urgent attention despite low heuristic scores?

**5. Overall Assessment:**
   A brief (2-3 paragraphs) summary of:
   - The overall health of the PR queue
   - Balance of simple vs complex PRs
   - Major themes or patterns you noticed
   - Top 3 priorities for maintainers

Please be specific, actionable, and technical. Consider the context of an ML compiler project
where correctness and performance are critical. Use the heuristic scores to guide your analysis,
but don't be afraid to disagree if you see something the heuristics missed."""

    return prompt


# Now merge the two analyses
def analyze_pr(
    pr_number: int, gdrive_files: list[str] | None = None, github_client: httpx.Client | None = None
) -> PRAnalysis:
    """
    Analyze a PR and return a PRAnalysis object.

    Args:
        pr_number: PR number to analyze
        gdrive_files: Optional list of Google Drive file names or URLs to use as context
                     e.g., ["ThunderQ4Plan", "ThunderBestPractices"]
    """
    print(f"Analyzing PR #{pr_number}...", file=sys.stderr)
    if gdrive_files:
        print(f"Using Google Drive files: {gdrive_files}", file=sys.stderr)

    # 1. Fetch all GitHub data
    pr = get_pr_data(pr_number, github_client=github_client)
    reviews = get_pr_reviews(pr_number, github_client=github_client)
    comments = get_pr_comments(pr_number, github_client=github_client)
    files = get_pr_files(pr_number, github_client=github_client)
    diff = get_pr_diff(pr_number, github_client=github_client)

    # 2. Calculate staleness
    days_open = calculate_days_diff(pr["created_at"])
    days_since_update = calculate_days_diff(pr["updated_at"])
    has_conflicts = pr["mergeable"] is False

    commits_behind = None
    try:
        comparison = compare_branches(pr["head"]["sha"], pr["base"]["ref"], github_client=github_client)
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

    # 3b. Assess internal Thunder team review status
    internal_review_status = assess_internal_review_status(reviews, comments)

    # 3c. Check Definition of Ready
    try:
        ci_checks = get_ci_check_runs(pr["head"]["sha"], github_client=github_client)
    except Exception as e:
        print(f"Warning: Could not fetch CI checks for PR #{pr_number}: {e}", file=sys.stderr)
        ci_checks = None

    definition_of_ready = check_definition_of_ready(pr, ci_checks)

    # 3d. Assess strategic goal alignment
    goals_manager = get_goals_manager()
    goal_alignment_obj = goals_manager.assess_pr_goal_alignment(pr)

    goal_alignment = (
        GoalAlignment(
            is_aligned=goal_alignment_obj.is_aligned,
            highest_priority=goal_alignment_obj.highest_priority,
            alignment_score=goal_alignment_obj.alignment_score,
            alignment_reasoning=goal_alignment_obj.alignment_reasoning,
            closed_issues=goal_alignment_obj.closed_issues,
        )
        if goal_alignment_obj
        else None
    )

    # 4. Run Heuristic Analysis
    heuristic_risk_score = assess_risk(pr, files, comments, reviews, staleness)
    heuristic_summary = generate_summary_heuristic(pr, files, comments, reviews)

    # 5. Assess complexity and impact
    complexity_score, _ = assess_complexity(pr, files)
    impact_score, _ = assess_impact(pr, heuristic_risk_score, review_status)

    # 6. Calculate Final Priority (complexity + impact + staleness matrix + external review boost + strategic goals)
    priority_score, priority_reasoning = calculate_priority(
        pr, heuristic_risk_score, staleness, review_status, files, internal_review_status, goal_alignment_obj
    )

    # 7. Prepare heuristic scores for LLM
    heuristic_scores = {
        "complexity": complexity_score,
        "impact": impact_score,
        "priority": priority_score,
        "priority_reasoning": priority_reasoning,
    }

    # 8. Run LLM Analysis - with heuristic scores and optional Google Drive context
    # This will print the prompt for this PR to stderr
    llm_results = run_llm_analysis(
        pr["number"], pr["title"], pr.get("body"), diff, gdrive_files=gdrive_files, heuristic_scores=heuristic_scores
    )

    # 9. Apply priority penalty for PRs that don't meet Definition of Ready
    adjusted_priority = priority_score
    priority_adjustment_reason = ""

    if not definition_of_ready.is_ready:
        # Apply penalty based on readiness score
        penalty = int((100 - definition_of_ready.readiness_score) / 5)  # Max penalty of 20 points
        adjusted_priority = max(0, priority_score - penalty)
        priority_adjustment_reason = f" (adjusted from {priority_score} due to Definition of Ready failures: {', '.join(definition_of_ready.failing_checks[:2])})"

    # 10. Combine all analysis
    analysis = PRAnalysis(
        number=pr["number"],
        title=pr["title"],
        author=pr["user"]["login"],
        created_at=pr["created_at"],
        updated_at=pr["updated_at"],
        url=pr["html_url"],
        labels=[label["name"] for label in pr["labels"]],
        summary=heuristic_summary,
        risk_score=heuristic_risk_score,
        priority_score=adjusted_priority,
        priority_reasoning=priority_reasoning + priority_adjustment_reason,
        complexity_score=complexity_score,
        impact_score=impact_score,
        llm_summary=llm_results["summary"],
        llm_risk_assessment=llm_results["risk_assessment"],
        llm_prompt=llm_results["llm_prompt"],  # Include the prompt
        staleness=staleness,
        review_status=review_status,
        internal_review_status=internal_review_status,  # Include internal team review tracking
        definition_of_ready=definition_of_ready,  # Include Definition of Ready assessment
        goal_alignment=goal_alignment,  # Include strategic goal alignment
        files_changed=len(files),
        additions=pr["additions"],
        deletions=pr["deletions"],
    )

    # 11. Add formatted display
    analysis.formatted_display = format_analysis_for_display(analysis)

    return analysis
