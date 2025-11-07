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
    priority_reasoning: str
    complexity_score: int
    impact_score: int
    staleness: StalenessInfo
    review_status: ReviewStatus
    files_changed: int
    additions: int
    deletions: int
    llm_summary: str
    llm_risk_assessment: str
