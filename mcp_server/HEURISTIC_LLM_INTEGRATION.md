# Heuristic + LLM Integration

This readme explains how the PR scoring works

## Overview

The PR review system integrates **heuristic analysis** with **LLM reasoning** to provide smart, context-aware guidance tailored to each PR's complexity and impact. The system includes comprehensive checks for:

- code quality,
- strategic alignment,
- team review status,
- and readiness for merge.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    PR Analysis Flow                             │
│                                                                 │
│  1. GitHub Data           →  PR details, diff, reviews, CI      │
│                                                                 │
│  2. Heuristic Analysis    →  Complexity (0-10)                  │
│                              Impact (0-10)                      │
│                              Risk Score (0-10)                  │
│                              Definition of Ready                │
│                              Internal Review Status             │
│                              Strategic Goal Alignment           │
│                                                                 │
│  3. Pass to LLM           →  LLM sees all heuristic scores      │
│                              + Original PR data                 |
│                              + Google Drive context (optional)  │
│                                                                 │
│  4. LLM Analysis          →  For SIMPLE PRs:                    │
│                              - Quick review checklist           │
│                              - Fast assessment                  │
│                                                                 |
│                              For COMPLEX PRs:                   │
│                              - Detailed review strategy         │
│                              - Debug checklist                  │
│                              - What could go wrong?             │
│                              - Testing recommendations          │
│                                                                 │
│  5. Combined Result       →  Heuristic scores + LLM insights    │
└─────────────────────────────────────────────────────────────────┘
```

## Heuristic Scoring System

### Priority Score (0-100)

The priority score is calculated from **4 weighted components**:

```
Priority = Base Score (50%) + Strategic (30%) + Review (15%) + Staleness (5%)
Maximum: 50 + 30 + 15 + 5 = 100
```

#### 1. Base Score (0-50 points)

Based on the **Complexity × Impact Matrix**:

| Complexity   | Impact    | Base Score | Category                                             |
| ------------ | --------- | ---------- | ---------------------------------------------------- |
| Simple (≤4)  | High (≥7) | **50**     | 🔥 **CRITICAL** - High impact, easy to review        |
| Simple (≤4)  | Low (\<7) | **40**     | ⚡ **QUICK WIN** - Easy to review and merge          |
| Complex (>4) | High (≥7) | **35**     | 🎯 **IMPORTANT** - High impact, needs careful review |
| Complex (>4) | Low (\<7) | **20**     | 📝 **LOW PRIORITY** - Complex with low impact        |

**Rationale:**

- **Simple + High Impact** gets highest priority (50) → Critical fixes that are easy to review
- **Simple + Low Impact** gets second priority (40) → Quick wins to clear the backlog
- **Complex + High Impact** gets medium-high (35) → Important but requires time
- **Complex + Low Impact** gets lowest (20) → Can be deferred

#### 2. Strategic Score (0-30 points)

Aligns PRs with **Q4 strategic goals**:

- **P0 (Must-Have):** +30 points 🔥
- **P1 (Should-Have):** +20 points 🎯
- **P2 (Nice-to-Have):** +10 points 📌
- **Not Aligned:** 0 points

**How it works:**

- PRs must link to a GitHub issue (e.g., `Closes #123`)
- Issues are associated with strategic goals via `link_issue_to_goal()`
- The highest priority goal is used for scoring

**Example:**

```python
# PR links to Issue #456
# Issue #456 is linked to "Q4-inference-opt" (P0 goal)
# → PR gets +30 strategic points
```

#### 3. Review Score (0-15 points)

Tracks internal and external review progress:

**Bonuses:**

- **+15 pts:** Ready for external review (2+ Thunder team approvals) ✅
- **+10 pts:** Approved and mergeable (alternative path)

**Penalties:**

- **-5 pts:** Changes requested
- **-10 pts:** Has merge conflicts

**Capped at:** 0-15 points

#### 4. Staleness Score (0-5 points)

Encourages merging old PRs, especially simple ones:

**Simple PRs (complexity ≤ 4):**

- **>90 days old:** +5 pts
- **>60 days old:** +3 pts
- **>30 days old:** +2 pts

**Complex PRs (complexity > 4):**

- **>90 days old:** +2 pts
- **>60 days old:** +1 pt
- **No activity in 30+ days:** -2 pts (likely stale)

**Capped at:** 0-5 points

### Complexity Score (0-10)

Measures how hard the PR is to review:

**Scoring factors:**

- **File count:**
  - \>20 files: +3
  - \>10 files: +2
  - \>5 files: +1
- **Lines changed:**
  - \>1000 lines: +3
  - \>500 lines: +2
  - \>100 lines: +1
- **Simple indicators:** "fix typo", "update doc", "formatting" → -2
- **Complex indicators:** "refactor", "architecture", "redesign" → +2
- **Core files:** >3 core/base/engine files → +2

**Categories:**

- **0-3:** 🟢 Simple (formatting, docs, small fixes)
- **4-6:** 🟡 Moderate (features, refactoring)
- **7-10:** 🔴 Complex (architecture changes, large refactors)

### Impact Score (0-10)

Measures the PR's importance to the project:

**Starting point:** 5 (medium impact)

**Adjustments:**

- **Security:** High (≥7) → +3, Medium (≥4) → +1
- **Urgency:** High (≥7) → +2, Medium (≥4) → +1
- **Breaking changes:** High (≥7) → +2
- **High-impact labels:** "critical", "blocker", "bug", "security", "performance" → +2
- **Low-impact labels:** "documentation", "style", "chore" → -2
- **Approved & mergeable:** +1

**Capped at:** 0-10

### Risk Score (0-10)

Multi-dimensional risk assessment:

#### Breaking Changes Risk (0-10)

- Keywords: "breaking", "deprecat", "remov", "incompatib" → +5
- API/interface files modified → +2
- \>20 files changed → +2
- "breaking" label → +5

#### Security Risk (0-10)

- Keywords: "security", "vulnerability", "auth", "credential" → +7
- Security-related files → +3
- "security" label → +8

#### Urgency Risk (0-10)

- Keywords: "block", "critical", "urgent", "hotfix", "bug" → +4
- \>90 days old → +3
- Critical labels → +5
- \>15 comments → +2 (high engagement)

**Overall risk:** Average of the three components

### Definition of Ready

Checks whether a PR meets the **Thunder Team PR Guidelines**:

#### Checklist (5 checks):

1. **✅ Descriptive Title**

   - Length ≥ 20 characters
   - Starts with capital letter
   - Example: "Add support for fused Adam optimizer"

1. **✅ Comprehensive Body**

   - Length ≥ 100 characters
   - Self-contained (no "per the title")
   - Explains what & why

1. **✅ Linked Issue**

   - Contains "closes #123", "fixes #456", or "resolves #789"
   - Exception: Small fixes (typos, formatting) can skip this

1. **✅ CI Passing**

   - All CI checks completed successfully
   - No failing tests

1. **✅ Not Draft**

   - PR is not marked as draft

**Scoring:**

- **Readiness Score:** (checks_passed / 5) × 100
- **80-100:** Ready with minor issues
- **60-79:** Needs attention
- **0-59:** Not ready

### Internal Review Status

Tracks **Thunder team** review progress per PR guidelines:

**Thunder Team Members:**

- crcrpar, kshitij12345, kiya00, riccardofelluga, beverlylytle, mattteochen, shino16

**Review Requirements:**

- **2 team approvals** required before pinging external maintainers
- **0 change requests** outstanding

**Status Messages:**

- ✅ "Ready - Can ping external maintainers (@lantiga, @t-vi, @KaelanDt)"
- ⏳ "Needs 1 more team approval (has 1/2)"
- ⏳ "Needs 2 team approvals (has 0/2)"
- 🔄 "Has X change request(s) from team - needs resolution"

**Impact on Priority:**

- Ready for external review → **+15 priority points**
- Changes requested → **-5 priority points**

### Strategic Goal Alignment

Links PRs to **Q4 strategic goals**:

**Goal Priorities:**

- **P0 (Must-Have):** Critical Q4 deliverables
- **P1 (Should-Have):** Important but not blocking
- **P2 (Nice-to-Have):** Future improvements

**How it works:**

1. Create strategic goals:

   ```python
   add_strategic_goal(
       goal_id="Q4-inference-opt",
       title="Inference Optimization",
       priority="P0",
       description="Optimize inference performance",
       theme="Performance",
   )
   ```

1. Link issues to goals:

   ```python
   link_issue_to_goal(issue_number=456, goal_id="Q4-inference-opt")
   ```

1. PRs linking to those issues get alignment score:

   - **P0:** +30 priority points 🔥
   - **P1:** +20 priority points 🎯
   - **P2:** +10 priority points 📌

**Benefit:** Ensures Q4 goals are prioritized in PR reviews!

## LLM Integration

### What the LLM Sees

The LLM receives a comprehensive prompt including:

```markdown
## HEURISTIC ANALYSIS

Our automated system has analyzed this PR:

**Complexity Score:** 8/10 (COMPLEX)
**Impact Score:** 9/10 (HIGH IMPACT)
**Priority Score:** 75/100

**Priority Reasoning:**
🎯 IMPORTANT (Complex + High Impact)
Complexity 8/10: 25 files changed, 2000 lines changed, refactors core files
Impact 9/10: high urgency, performance label, high-impact label
Strategic: 🔥 P0 STRATEGIC GOAL (closes #456) (+30pts)
Review: ✅ ready for external review (2+ team approvals) (+15pts)
Staleness: aging 65 days (+2pts)
Final: 82/100 (Base:35 + Strategic:30 + Review:15 + Staleness:2)

**Review Guidance:** This is a complex change. Pay special attention to
architecture, testing, and potential side effects.

---

## DEFINITION OF READY
**Readiness Score:** 80/100
**Status:** ⚠️ Not ready - 1 check(s) failing

**Failing Checks:**
- ❌ CI checks failing: test_distributed.yaml

---

## INTERNAL REVIEW STATUS
✅ Ready - Can ping external maintainers (@lantiga, @t-vi, @KaelanDt)
- Team Approvals: 2/2
- Reviewers: kshitij12345, mattteochen

---

## STRATEGIC GOAL ALIGNMENT
**Aligned:** Yes 🔥
**Priority:** P0 (Must-Have)
**Goal:** Q4 Inference Optimization
**Closes:** #456
```

### LLM Response Sections

#### For SIMPLE PRs (complexity < 7):

**2 Sections:**

1. **Summary** - What and why
1. **Risk Assessment** - Breaking changes, security, urgency

#### For COMPLEX PRs (complexity ≥ 7):

**3 Sections:**

1. **Summary** - What and why
1. **Risk Assessment** - Breaking changes, security, urgency
1. **Review Checklist & Debugging Guide** ✨
   - Key areas to review
   - Potential issues
   - Testing strategy
   - Architecture impact
   - Debug checklist
