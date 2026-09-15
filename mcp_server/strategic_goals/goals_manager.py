import re
from dataclasses import dataclass
from typing import Any


@dataclass
class StrategicGoal:
    """Represents a strategic goal (e.g., Q4 priority)"""

    id: str  # e.g., "Q4-inference-optimization"
    title: str  # e.g., "Inference Optimization"
    priority: str  # "P0", "P1", "P2"
    description: str
    linked_issues: list[int]  # GitHub issue numbers
    theme: str  # e.g., "Performance", "Features", "Technical Debt"


@dataclass
class GoalAlignment:
    """Represents how a PR aligns with strategic goals"""

    is_aligned: bool
    linked_goals: list[StrategicGoal]
    highest_priority: str | None  # "P0", "P1", "P2"
    closed_issues: list[int]  # Issues closed by this PR
    alignment_score: int  # 0-100, how well PR aligns with goals
    alignment_reasoning: str


class StrategicGoalsManager:
    """
    Manages strategic goals and their alignment with PRs.

    Goals can be defined:
    1. From Google Drive documents (Q4 plans, roadmaps)
    2. Manual configuration
    3. Extracted from GitHub milestones/projects
    """

    def __init__(self):
        self.goals: dict[str, StrategicGoal] = {}
        self.issue_to_goal_map: dict[int, list[str]] = {}  # issue_number -> [goal_ids]

    def add_goal(self, goal: StrategicGoal):
        """Add a strategic goal to the manager"""
        self.goals[goal.id] = goal

        # Update reverse mapping (issue -> goals)
        for issue_num in goal.linked_issues:
            if issue_num not in self.issue_to_goal_map:
                self.issue_to_goal_map[issue_num] = []
            if goal.id not in self.issue_to_goal_map[issue_num]:
                self.issue_to_goal_map[issue_num].append(goal.id)

    def link_issue_to_goal(self, issue_number: int, goal_id: str):
        """Manually link an issue to a goal"""
        if goal_id in self.goals:
            if issue_number not in self.goals[goal_id].linked_issues:
                self.goals[goal_id].linked_issues.append(issue_number)

            if issue_number not in self.issue_to_goal_map:
                self.issue_to_goal_map[issue_number] = []
            if goal_id not in self.issue_to_goal_map[issue_number]:
                self.issue_to_goal_map[issue_number].append(goal_id)

    def extract_closed_issues(self, pr_body: str | None) -> list[int]:
        """
        Extract issue numbers that the PR closes.

        Looks for patterns like:
        - Closes #123
        - Fixes #456
        - Resolves #789
        """
        if not pr_body:
            return []

        patterns = [
            r"closes\s+#(\d+)",
            r"close\s+#(\d+)",
            r"fixes\s+#(\d+)",
            r"fix\s+#(\d+)",
            r"resolves\s+#(\d+)",
            r"resolve\s+#(\d+)",
        ]

        closed_issues = set()
        pr_body_lower = pr_body.lower()

        for pattern in patterns:
            matches = re.finditer(pattern, pr_body_lower)
            for match in matches:
                issue_num = int(match.group(1))
                closed_issues.add(issue_num)

        return sorted(list(closed_issues))

    def assess_pr_goal_alignment(self, pr: dict[str, Any]) -> GoalAlignment:
        """
        Assess how well a PR aligns with strategic goals.

        Args:
            pr: PR data from GitHub API

        Returns:
            GoalAlignment with alignment details and scoring
        """
        pr_body = pr.get("body") or ""

        # Extract issues closed by this PR
        closed_issues = self.extract_closed_issues(pr_body)

        # Find which goals these issues are linked to
        linked_goal_ids = set()
        for issue_num in closed_issues:
            if issue_num in self.issue_to_goal_map:
                linked_goal_ids.update(self.issue_to_goal_map[issue_num])

        linked_goals = [self.goals[gid] for gid in linked_goal_ids if gid in self.goals]

        # Determine highest priority
        highest_priority = None
        if linked_goals:
            priorities = [g.priority for g in linked_goals]
            if "P0" in priorities:
                highest_priority = "P0"
            elif "P1" in priorities:
                highest_priority = "P1"
            elif "P2" in priorities:
                highest_priority = "P2"

        # Calculate alignment score (0-100)
        alignment_score = 0
        reasoning_parts = []

        if not linked_goals:
            # Not aligned with any strategic goal
            alignment_score = 0
            reasoning_parts.append("Not linked to any strategic goals")
        else:
            # Aligned with strategic goals
            if highest_priority == "P0":
                alignment_score = 100
                reasoning_parts.append(f"🎯 STRATEGIC: Linked to {len(linked_goals)} P0 goal(s)")
            elif highest_priority == "P1":
                alignment_score = 75
                reasoning_parts.append(f"🎯 HIGH: Linked to {len(linked_goals)} P1 goal(s)")
            elif highest_priority == "P2":
                alignment_score = 50
                reasoning_parts.append(f"🎯 MEDIUM: Linked to {len(linked_goals)} P2 goal(s)")

            # List specific goals
            for goal in linked_goals:
                reasoning_parts.append(
                    f"  - {goal.priority}: {goal.title} (closes #{', #'.join(map(str, [i for i in closed_issues if i in goal.linked_issues]))})"
                )

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else "No strategic alignment"

        return GoalAlignment(
            is_aligned=len(linked_goals) > 0,
            linked_goals=linked_goals,
            highest_priority=highest_priority,
            closed_issues=closed_issues,
            alignment_score=alignment_score,
            alignment_reasoning=reasoning,
        )

    def calculate_strategic_priority_boost(self, alignment: GoalAlignment) -> tuple[int, str]:
        """
        Calculate priority boost based on strategic goal alignment.

        P0 goals: +50 points (massive boost)
        P1 goals: +30 points (significant boost)
        P2 goals: +15 points (moderate boost)

        Args:
            alignment: GoalAlignment from assess_pr_goal_alignment()

        Returns:
            (boost_points, reasoning)
        """
        if not alignment.is_aligned:
            return 0, "No strategic alignment"

        if alignment.highest_priority == "P0":
            boost = 50
            reasoning = f"🔥 P0 STRATEGIC: {len(alignment.linked_goals)} goal(s) - MAXIMUM PRIORITY"
        elif alignment.highest_priority == "P1":
            boost = 30
            reasoning = f"🎯 P1 HIGH: {len(alignment.linked_goals)} goal(s) - High strategic priority"
        elif alignment.highest_priority == "P2":
            boost = 15
            reasoning = f"📌 P2 MEDIUM: {len(alignment.linked_goals)} goal(s) - Moderate strategic priority"
        else:
            boost = 0
            reasoning = "Unknown priority level"

        return boost, reasoning

    def clear_goals(self):
        """Clear all goals and mappings"""
        self.goals.clear()
        self.issue_to_goal_map.clear()


# Global instance
_goals_manager = StrategicGoalsManager()


def get_goals_manager() -> StrategicGoalsManager:
    """Get the global goals manager instance"""
    return _goals_manager
