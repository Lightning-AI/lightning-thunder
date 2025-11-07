from typing import Any
import re
from pr_scores.scores import (RiskScore, StalenessInfo, ReviewStatus, InternalReviewStatus, DefinitionOfReadyStatus, RiskReasoning)


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


def assess_complexity(pr: dict[str, Any], files: list[dict[str, Any]]) -> tuple[int, str]:
    """
    Assess PR complexity (0-10).

    Returns:
        (complexity_score, reasoning)
        0-3: Simple (formatting, docs, small fixes)
        4-6: Moderate (feature additions, refactoring)
        7-10: Complex (architecture changes, large refactors)
    """
    complexity = 0
    reasons = []

    # File count
    file_count = len(files)
    if file_count > 20:
        complexity += 3
        reasons.append(f"{file_count} files changed")
    elif file_count > 10:
        complexity += 2
        reasons.append(f"{file_count} files changed")
    elif file_count > 5:
        complexity += 1

    # Lines changed
    total_changes = pr["additions"] + pr["deletions"]
    if total_changes > 1000:
        complexity += 3
        reasons.append(f"{total_changes} lines changed")
    elif total_changes > 500:
        complexity += 2
        reasons.append(f"{total_changes} lines changed")
    elif total_changes > 100:
        complexity += 1

    # Check if it's mostly simple changes
    title_lower = pr["title"].lower()
    body_lower = (pr["body"] or "").lower()

    simple_indicators = ["fix typo", "update doc", "formatting", "style", "comment", "readme"]
    complex_indicators = ["refactor", "architecture", "redesign", "rewrite", "migration"]

    if any(ind in title_lower or ind in body_lower for ind in simple_indicators):
        complexity = max(0, complexity - 2)
        reasons.append("simple change indicators")

    if any(ind in title_lower or ind in body_lower for ind in complex_indicators):
        complexity += 2
        reasons.append("complex change indicators")

    # Check file types
    core_files = [f for f in files if any(x in f["filename"].lower() for x in ["core/", "base", "engine"])]
    if len(core_files) > 3:
        complexity += 2
        reasons.append("modifies core files")

    complexity = min(10, max(0, complexity))
    reasoning = f"Complexity {complexity}/10: {', '.join(reasons) if reasons else 'standard changes'}"

    return complexity, reasoning


def assess_impact(pr: dict[str, Any], risk: RiskScore, review_status: ReviewStatus) -> tuple[int, str]:
    """
    Assess PR impact on project (0-10).

    Returns:
        (impact_score, reasoning)
        0-3: Low impact (minor improvements, non-critical)
        4-6: Medium impact (features, improvements)
        7-10: High impact (critical fixes, major features, security)
    """
    impact = 5  # Start at medium
    reasons = []

    # Security and urgency are high impact
    if risk.security >= 7:
        impact += 3
        reasons.append("security critical")
    elif risk.security >= 4:
        impact += 1
        reasons.append("security related")

    if risk.urgency_if_not_merged >= 7:
        impact += 2
        reasons.append("high urgency")
    elif risk.urgency_if_not_merged >= 4:
        impact += 1

    # Breaking changes = high impact
    if risk.breaking_changes >= 7:
        impact += 2
        reasons.append("breaking changes")

    # Check labels for impact indicators
    labels_lower = [label["name"].lower() for label in pr.get("labels", [])]

    high_impact_labels = ["critical", "blocker", "bug", "security", "performance"]
    low_impact_labels = ["documentation", "style", "chore", "refactor"]

    if any(label in labels_lower for label in high_impact_labels):
        impact += 2
        reasons.append("high-impact label")

    if any(label in labels_lower for label in low_impact_labels):
        impact -= 2
        reasons.append("low-impact label")

    # Approved PRs ready to merge = high impact (unblocking work)
    if review_status.approved_reviews > 0 and pr.get("mergeable"):
        impact += 1
        reasons.append("approved and ready")

    impact = min(10, max(0, impact))
    reasoning = f"Impact {impact}/10: {', '.join(reasons) if reasons else 'standard impact'}"

    return impact, reasoning


def calculate_priority(
    pr: dict[str, Any],
    risk: RiskScore,
    staleness: StalenessInfo,
    review_status: ReviewStatus,
    files: list[dict[str, Any]],
    internal_review_status: InternalReviewStatus | None = None,
    goal_alignment: Any | None = None,  # GoalAlignment from strategic_goals
) -> tuple[int, str]:
    """
    Calculate priority score (0-100) based on complexity, impact, staleness, internal review, and strategic goals.

    Priority Matrix:
    - Easy + High Impact → VERY HIGH (90-100)
    - Easy + Low Impact → HIGH (70-89) - Quick wins
    - Complex + High Impact → MEDIUM-HIGH (60-79) - Needs careful review
    - Complex + Low Impact → LOW (0-59) - Deprioritize

    Boosts:
    - Staleness: Simple stale PRs get priority bump
    - External Review: PRs ready for external review get +25
    - Strategic Goals: P0 goals get +50, P1 get +30, P2 get +15

    Returns:
        (priority_score, reasoning)
    """
    # Assess complexity and impact
    complexity, complexity_reason = assess_complexity(pr, files)
    impact, impact_reason = assess_impact(pr, risk, review_status)

    # Categorize
    is_simple = complexity <= 4
    is_high_impact = impact >= 7

    # Base score from matrix
    if is_simple and is_high_impact:
        # Easy + Huge Impact → VERY HIGH PRIORITY
        base_score = 90
        category = "🔥 CRITICAL (Simple + High Impact)"
    elif is_simple and not is_high_impact:
        # Easy + Small Impact → HIGH PRIORITY (Quick wins)
        base_score = 75
        category = "⚡ QUICK WIN (Simple + Low Impact)"
    elif not is_simple and is_high_impact:
        # Complex + Huge Impact → MEDIUM-HIGH (Needs careful review)
        base_score = 65
        category = "🎯 IMPORTANT (Complex + High Impact)"
    else:
        # Complex + Small Impact → LOW PRIORITY
        base_score = 40
        category = "📝 LOW (Complex + Low Impact)"

    # Staleness adjustments
    staleness_adjustment = 0
    staleness_reasons = []

    if staleness.days_open > 90:
        if is_simple:
            # Simple stale PR → BIG BOOST (get it done!)
            staleness_adjustment += 15
            staleness_reasons.append(f"stale {staleness.days_open} days (simple PR boost)")
        else:
            # Complex stale PR → smaller boost
            staleness_adjustment += 5
            staleness_reasons.append(f"stale {staleness.days_open} days")
    elif staleness.days_open > 60:
        if is_simple:
            staleness_adjustment += 10
            staleness_reasons.append(f"aging {staleness.days_open} days (simple PR boost)")
        else:
            staleness_adjustment += 3
            staleness_reasons.append(f"aging {staleness.days_open} days")
    elif staleness.days_open > 30:
        if is_simple:
            staleness_adjustment += 5
            staleness_reasons.append(f"waiting {staleness.days_open} days")

    # Penalties
    if staleness.has_conflicts:
        staleness_adjustment -= 20
        staleness_reasons.append("has conflicts")

    if review_status.changes_requested > 0:
        staleness_adjustment -= 10
        staleness_reasons.append("changes requested")

    if staleness.days_since_update > 30 and not is_simple:
        # Complex PR with no recent activity → deprioritize
        staleness_adjustment -= 10
        staleness_reasons.append("no recent activity")

    # Bonuses
    if review_status.approved_reviews > 0 and staleness.is_mergeable:
        staleness_adjustment += 15
        staleness_reasons.append("approved and mergeable")

    # SIGNIFICANT BOOST: PR is ready for external review (2+ team approvals)
    # This ensures PRs that have passed internal review are actioned quickly
    if internal_review_status and internal_review_status.is_ready_for_external_review:
        staleness_adjustment += 25
        staleness_reasons.append("✅ ready for external review (2+ team approvals)")

    # STRATEGIC GOAL BOOST: PR aligns with Q4/strategic goals
    # This HEAVILY weights PRs that contribute to P0/P1/P2 goals
    strategic_boost = 0
    if goal_alignment and goal_alignment.is_aligned:
        if goal_alignment.highest_priority == "P0":
            strategic_boost = 50
            staleness_reasons.append(
                f"🔥 P0 STRATEGIC GOAL (closes #{', #'.join(map(str, goal_alignment.closed_issues))})"
            )
        elif goal_alignment.highest_priority == "P1":
            strategic_boost = 30
            staleness_reasons.append(
                f"🎯 P1 HIGH PRIORITY GOAL (closes #{', #'.join(map(str, goal_alignment.closed_issues))})"
            )
        elif goal_alignment.highest_priority == "P2":
            strategic_boost = 15
            staleness_reasons.append(
                f"📌 P2 MEDIUM PRIORITY GOAL (closes #{', #'.join(map(str, goal_alignment.closed_issues))})"
            )

    # Final score (strategic boost applied AFTER other adjustments for maximum impact)
    final_score = base_score + staleness_adjustment + strategic_boost
    final_score = max(0, min(100, int(final_score)))

    # Build reasoning
    reasoning_parts = [
        category,
        complexity_reason,
        impact_reason,
    ]
    if staleness_reasons:
        reasoning_parts.append(f"Staleness: {', '.join(staleness_reasons)}")
    reasoning_parts.append(f"Final: {final_score}/100")

    reasoning = "\n".join(reasoning_parts)

    return final_score, reasoning


def assess_internal_review_status(
    reviews: list[dict[str, Any]],
    comments: list[dict[str, Any]],
) -> InternalReviewStatus:
    """
    Assess internal Thunder team review status according to PR guidelines.

    Per Thunder Team PR Guidelines:
    - PRs should get 2 internal team approvals before pinging external maintainers
    - Team members: crcrpar, kshitij12345, kiya00, riccardofelluga, beverlylytle, mattteochen, shino16

    Args:
        reviews: List of PR reviews from GitHub
        comments: List of PR comments from GitHub

    Returns:
        InternalReviewStatus with team approval tracking
    """
    # Thunder team member GitHub handles
    THUNDER_TEAM = {"crcrpar", "kshitij12345", "kiya00", "riccardofelluga", "beverlylytle", "mattteochen", "shino16"}

    # Track team reviews
    team_approvals = 0
    team_changes_requested = 0
    team_reviewers = set()

    # Process all reviews
    for review in reviews:
        reviewer = review["user"]["login"]
        state = review["state"]

        if reviewer in THUNDER_TEAM:
            team_reviewers.add(reviewer)
            if state == "APPROVED":
                team_approvals += 1
            elif state == "CHANGES_REQUESTED":
                team_changes_requested += 1

    # Determine if ready for external review per guidelines
    is_ready = team_approvals >= 2 and team_changes_requested == 0

    # Build status message
    if team_approvals >= 2 and team_changes_requested == 0:
        status_msg = "✅ Ready - Can ping external maintainers (@lantiga, @t-vi, @KaelanDt)"
    elif team_approvals == 1 and team_changes_requested == 0:
        status_msg = f"⏳ Needs 1 more team approval (has {team_approvals}/2)"
    elif team_approvals == 0:
        status_msg = "⏳ Needs 2 team approvals (has 0/2)"
    elif team_changes_requested > 0:
        status_msg = f"🔄 Has {team_changes_requested} change request(s) from team - needs resolution"
    else:
        status_msg = f"⏳ Has {team_approvals}/2 approvals, {team_changes_requested} change requests"

    return InternalReviewStatus(
        thunder_team_approvals=team_approvals,
        thunder_team_changes_requested=team_changes_requested,
        thunder_team_reviewers=sorted(list(team_reviewers)),
        is_ready_for_external_review=is_ready,
        review_guideline_status=status_msg,
    )


def check_definition_of_ready(
    pr: dict[str, Any], ci_checks: list[dict[str, Any]] | None = None
) -> DefinitionOfReadyStatus:
    """
    Assess whether a PR meets the Definition of Ready checklist per Thunder Team guidelines.

    Checks:
    - Atomic & Focused: Inferred from file count and changes
    - Descriptive Title: Length >= 20 chars, starts with capital letter
    - Comprehensive Body: Length >= 100 chars, includes issue link or explanation
    - Linked Issue: Contains "closes #", "fixes #", "resolves #", etc.
    - CI Passing: All CI checks are passing (or at least not failing)
    - Not Draft: PR is not marked as draft

    Args:
        pr: PR data from GitHub API
        ci_checks: Optional list of CI check runs for the PR's head commit

    Returns:
        DefinitionOfReadyStatus with readiness assessment
    """
    failing_checks = []

    # 1. Check if PR is a draft
    is_draft = pr.get("draft", False)
    if is_draft:
        failing_checks.append("PR is marked as draft")

    # 2. Check for descriptive title
    title = pr.get("title", "")
    has_descriptive_title = True

    if len(title) < 20:
        has_descriptive_title = False
        failing_checks.append(f"Title too short ({len(title)} chars, need >= 20)")
    elif not title[0].isupper():
        has_descriptive_title = False
        failing_checks.append("Title should start with capital letter")

    # 3. Check for comprehensive body
    body = pr.get("body") or ""
    has_comprehensive_body = True

    if len(body.strip()) < 100:
        has_comprehensive_body = False
        failing_checks.append(f"Body too short ({len(body.strip())} chars, need >= 100)")
    elif body.strip().lower() == title.lower():
        has_comprehensive_body = False
        failing_checks.append("Body just repeats the title - needs explanation")
    elif "per the title" in body.lower() or "see title" in body.lower():
        has_comprehensive_body = False
        failing_checks.append("Body references title instead of being self-contained")

    # 4. Check for linked issue
    has_linked_issue = False
    issue_link_patterns = [
        r"closes\s+#\d+",
        r"fixes\s+#\d+",
        r"resolves\s+#\d+",
        r"fix\s+#\d+",
        r"close\s+#\d+",
        r"resolve\s+#\d+",
    ]

    body_lower = body.lower()
    for pattern in issue_link_patterns:
        if re.search(pattern, body_lower):
            has_linked_issue = True
            break

    # Note: Not having a linked issue is not always a failure (e.g., for small fixes)
    # So we track it but don't always penalize
    if not has_linked_issue:
        # Check if it's explained as a small fix or improvement
        small_fix_indicators = ["typo", "small fix", "minor", "quick fix", "formatting"]
        is_explained_small_fix = any(indicator in body_lower for indicator in small_fix_indicators)

        if not is_explained_small_fix and len(body.strip()) > 0:
            # Only fail if it's not obviously a small fix
            failing_checks.append("No linked issue (closes #, fixes #, resolves #)")

    # 5. Check CI status
    ci_passing = True
    if ci_checks is not None:
        # If we have CI check data, analyze it
        if len(ci_checks) == 0:
            # No CI checks configured - consider it passing
            ci_passing = True
        else:
            # Check if all required checks are passing
            failing_ci = []
            pending_ci = []

            for check in ci_checks:
                status = check.get("status")  # "queued", "in_progress", "completed"
                conclusion = check.get("conclusion")  # "success", "failure", "neutral", "cancelled", etc.

                if status == "completed":
                    if conclusion not in ["success", "neutral", "skipped"]:
                        failing_ci.append(check.get("name", "unknown"))
                elif status in ["queued", "in_progress"]:
                    pending_ci.append(check.get("name", "unknown"))

            if failing_ci:
                ci_passing = False
                failing_checks.append(f"CI checks failing: {', '.join(failing_ci[:3])}")
            elif pending_ci:
                # Pending CI is not a hard failure, but worth noting
                ci_passing = True  # Don't block on pending
    else:
        # No CI check data provided - check mergeable_state as fallback
        mergeable_state = pr.get("mergeable_state", "unknown")
        if mergeable_state in ["dirty", "blocked"]:
            ci_passing = False
            failing_checks.append(f"PR mergeable state: {mergeable_state}")

    # Calculate overall readiness
    checks_passed = sum(
        [
            not is_draft,
            has_descriptive_title,
            has_comprehensive_body,
            has_linked_issue or "no linked issue" not in [c.lower() for c in failing_checks],
            ci_passing,
        ]
    )
    total_checks = 5
    readiness_score = int((checks_passed / total_checks) * 100)

    is_ready = len(failing_checks) == 0 and not is_draft

    # Build summary message
    if is_ready:
        readiness_summary = f"✅ Ready for review ({checks_passed}/{total_checks} checks passed)"
    elif is_draft:
        readiness_summary = "🚧 Draft PR - not ready for review"
    else:
        readiness_summary = f"⚠️ Not ready - {len(failing_checks)} check(s) failing"

    return DefinitionOfReadyStatus(
        has_linked_issue=has_linked_issue,
        has_descriptive_title=has_descriptive_title,
        has_comprehensive_body=has_comprehensive_body,
        ci_passing=ci_passing,
        is_draft=is_draft,
        is_ready=is_ready,
        failing_checks=failing_checks,
        readiness_score=readiness_score,
        readiness_summary=readiness_summary,
    )
