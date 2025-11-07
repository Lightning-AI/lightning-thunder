from dataclasses import dataclass


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
class InternalReviewStatus:
    """Tracks internal Thunder team reviews per PR guidelines"""
    thunder_team_approvals: int
    thunder_team_changes_requested: int
    thunder_team_reviewers: list[str]
    is_ready_for_external_review: bool
    review_guideline_status: str


@dataclass
class DefinitionOfReadyStatus:
    """Tracks whether a PR meets the Definition of Ready checklist"""
    has_linked_issue: bool
    has_descriptive_title: bool
    has_comprehensive_body: bool
    ci_passing: bool
    is_draft: bool
    is_ready: bool  # Overall readiness (all checks pass)
    failing_checks: list[str]  # List of checks that failed
    readiness_score: int  # 0-100 score based on checks passed
    readiness_summary: str  # Human-readable summary


@dataclass
class GoalAlignment:
    """Strategic goal alignment (imported from strategic_goals module)"""
    is_aligned: bool
    highest_priority: str | None
    alignment_score: int
    alignment_reasoning: str
    closed_issues: list[int]


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
    priority_reasoning: str
    complexity_score: int
    impact_score: int
    staleness: StalenessInfo
    review_status: ReviewStatus
    internal_review_status: InternalReviewStatus
    definition_of_ready: DefinitionOfReadyStatus
    goal_alignment: GoalAlignment | None  # Strategic goal alignment
    files_changed: int
    additions: int
    deletions: int
    llm_summary: str
    llm_risk_assessment: str
    llm_prompt: str  # The full LLM prompt for easy re-processing
