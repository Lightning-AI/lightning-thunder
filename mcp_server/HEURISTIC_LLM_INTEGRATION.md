# Heuristic + LLM Integration

## Overview

The PR review system now integrates **heuristic analysis** with **LLM reasoning** to provide smart, context-aware guidance tailored to each PR's complexity and impact.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    PR Analysis Flow                             │
│                                                                 │
│  1. GitHub Data           →  PR details, diff, reviews          │
│                                                                 │
│  2. Heuristic Analysis    →  Complexity (0-10)                  │
│                              Impact (0-10)                      │
│                              Priority (0-100)                   |
│                              Priority Reasoning                 │
│                                                                 │
│  3. Pass to LLM           →  LLM sees heuristic scores          │
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

## Single PR Analysis

### What the LLM Sees

```
## HEURISTIC ANALYSIS

Our automated system has analyzed this PR:

**Complexity Score:** 8/10 (COMPLEX)
**Impact Score:** 9/10 (HIGH IMPACT)
**Priority Score:** 75/100

**Priority Reasoning:**
🎯 IMPORTANT (Complex + High Impact)
Complexity 8/10: 25 files changed, 2000 lines changed, refactors core files
Impact 9/10: high urgency, performance label, high-impact label
Staleness: aging 65 days
Final: 75/100

**Review Guidance:** This is a complex change. Pay special attention to
architecture, testing, and potential side effects.
```

### LLM Response for COMPLEX PR

For complex PRs (complexity ≥ 7), the LLM provides **THREE sections**:

1. **Summary** - What and why
1. **Risk Assessment** - Breaking changes, security, urgency
1. **Review Checklist & Debugging Guide** - NEW! ✨
   - Key areas to review
   - Potential issues
   - Testing strategy
   - Architecture impact
   - Debug checklist

### LLM Response for SIMPLE PR

For simple/moderate PRs (complexity < 7), the LLM provides **TWO sections**:

1. **Summary** - What and why
1. **Risk Assessment** - Breaking changes, security, urgency

## Example: Complex PR

### Heuristic Analysis

```
Complexity: 8/10 (COMPLEX)
- 25 files changed
- 2000 lines changed
- Refactors core execution engine
- Contains "refactor" keyword

Impact: 9/10 (HIGH IMPACT)
- Performance critical
- Approved by 2 reviewers
- High urgency label

Priority: 75/100 (🎯 IMPORTANT)
```

### LLM Sees This + Code

The LLM gets:

- All heuristic scores above
- PR description
- Code diff
- Review guidance: "This is complex - be extra careful"

### LLM Provides

```markdown
**Summary:**
This PR refactors the core execution engine to improve performance by 40%...

###

**Risk Assessment:**
- Breaking Changes: Medium risk - API signatures remain the same but
  internal behavior changes significantly...
- Security: Low risk - no new attack surfaces...
- Urgency: High - blocking Q4 performance goals...

###

**Review Checklist & Debugging Guide:**

**Key Areas to Review:**
1. Execution engine state management (lines 450-680)
2. Memory allocation patterns in optimizer.py
3. Thread safety in parallel execution paths

**Potential Issues:**
- Memory leaks if caching isn't properly cleared
- Race conditions in multi-threaded scenarios
- Performance regression for single-threaded workloads

**Testing Strategy:**
- Unit tests for all execution paths
- Stress tests with concurrent workloads
- Benchmark suite comparing old vs new engine
- Edge cases: empty inputs, very large tensors

**Architecture Impact:**
- Changes fundamental execution model
- Affects all downstream consumers of execution engine
- May require updates to documentation

**Debug Checklist (if issues arise):**
1. Check memory profiler for leaks
2. Verify thread safety with ThreadSanitizer
3. Compare execution traces old vs new
4. Test with single-threaded mode first
5. Check optimizer cache invalidation logic
```

## Batch Analysis

### What the LLM Sees for Each PR

```
### PR #123: Refactor execution engine

**Heuristic Analysis:**
- Priority Score: 75/100
- 🔴 Complexity: 8/10 (COMPLEX) - 25 files changed, 2000 lines, refactors core files
- 🔴 Impact: 9/10 (HIGH) - high urgency, performance label
- Priority Reasoning:
  🎯 IMPORTANT (Complex + High Impact)
  Complexity 8/10: refactors core files
  Impact 9/10: performance critical
  Staleness: aging 65 days
  Final: 75/100
```

### LLM Task Instructions

The LLM is asked to:

**For SIMPLE PRs (🟢):**

- Quick review checklist
- Estimated review time
- Can it be merged quickly?

**For MODERATE PRs (🟡):**

- Key areas to focus on
- Integration concerns
- Testing recommendations

**For COMPLEX PRs (🔴):**

- Detailed review strategy
- What could go wrong?
- Debug checklist
- Should it be broken into smaller PRs?

### LLM Output

```markdown
**1. Enhanced Priority Assessment:**

PR #123: Refactor execution engine (🔴 COMPLEX, 🔴 HIGH IMPACT)
- Review Strategy: Schedule 4-hour deep review session
- Risk Areas: State management, thread safety, memory allocation
- Testing Must-Cover: Concurrent workloads, memory profiling, benchmarks
- Recommendation: This is appropriate scope - don't split
- Debug Checklist: Memory profiler, ThreadSanitizer, execution traces

PR #456: Fix typo in docs (🟢 SIMPLE, 🔵 LOW IMPACT)
- Quick Checklist: Verify typo is actually wrong, check rendering
- Review Time: 2 minutes
- Can Merge: Yes, immediately after approval

**2. Prioritized Review Order:**

🔥 CRITICAL (Review Today):
- PR #789: Security fix (simple, 3 files)

⚡ QUICK WINS (Clear This Week):
- PR #456: Typo fix (2 min review)
- PR #567: Update changelog (5 min review)

🎯 IMPORTANT (Schedule Deep Review):
- PR #123: Execution engine refactor (4 hours needed)
- PR #234: Performance optimization (2 hours needed)

**3. Complexity-Specific Guidance:**

PR #123 (COMPLEX):
- What makes it complex: Touches 25 core files, changes execution model
- Extra careful on: Thread safety, memory management, API compatibility
- Must test: Concurrent scenarios, memory leaks, performance regression
- Simplification: Already well-scoped, don't split

...
```

## Benefits

### ✅ Context-Aware Guidance

**Simple PR:** "Quick review - check spelling, verify rendering, merge"
**Complex PR:** "Deep review needed - here's a 5-point debug checklist"

### ✅ Leverages Both Strengths

**Heuristics:** Fast, consistent, measurable (complexity/impact scores)
**LLM:** Nuanced understanding, contextual reasoning, specific guidance

### ✅ Actionable for Reviewers

Instead of: "This is complex"
You get: "This is complex BECAUSE of X, Y, Z. Watch out for A, B, C. Test P, Q, R."

### ✅ Prevents Analysis Paralysis

**For complex PRs:** LLM breaks down what to check
**For simple PRs:** LLM confirms it's simple and safe to merge

## Usage

### Single PR Analysis

```python
# Analyze with heuristic integration
analyze_single_pr(pr_number=123, gdrive_files=["ThunderBestPractices"])  # Optional

# If complexity ≥ 7, you'll get detailed debugging guidance
# If complexity < 7, you'll get standard analysis
```

### Batch Analysis

```python
# Batch analyze with heuristic integration
llm_batch_analysis(
    min_priority=50, limit=10, gdrive_files=["ThunderQ4Plan"]  # Optional
)

# LLM will provide tailored guidance for each PR
# based on its complexity/impact category
```

### Customize Guidance Content

Edit the prompt templates in `run_llm_analysis()`:

```python
# For complex PRs, add/modify sections:
**Review Checklist & Debugging Guide:**
-   **Key Areas to Review:** ...
-   **Potential Issues:** ...
-   **Testing Strategy:** ...
-   **Your Custom Section:** ...
```

## Example Workflow

### Morning PR Triage

```python
# 1. Quick triage with heuristics
result = prioritize_prs(min_priority=70)

# Shows you:
# - 5 simple PRs (quick wins) ⚡
# - 2 complex PRs (need deep review) 🎯
# - 3 critical PRs (do today) 🔥

# 2. Knock out simple PRs first
for pr in simple_prs:
    analyze_single_pr(pr)  # Gets quick checklist from LLM
    # Review takes 5-10 minutes each

# 3. Schedule complex PRs
for pr in complex_prs:
    analysis = analyze_single_pr(pr)  # Gets detailed debug guide
    # Schedule 2-4 hour review session
    # Use LLM's debug checklist during review
```

## Real-World Example

### PR: Large Refactor (30 files, 3000 lines)

**Heuristic:** Complexity 9/10, Impact 8/10, Priority 72/100

**LLM Analysis:**

```
This PR refactors the distributed training coordinator. Here's your review plan:

**Key Areas (focus here):**
1. State synchronization logic (coordinator.py:450-680)
2. Error handling in failure scenarios
3. Network protocol changes

**What Could Go Wrong:**
- Race conditions during coordinator election
- Data loss if crash during state sync
- Performance degradation with 100+ workers

**Testing Strategy (must cover):**
- Coordinator failure during training
- Network partition scenarios
- Scale test: 1, 10, 100, 1000 workers
- Crash recovery from checkpoints

**Debug Checklist (if production issues):**
1. Check coordinator election logs
2. Verify state sync timing
3. Test network timeout settings
4. Review checkpoint validity
5. Check worker heartbeat mechanism
```

**Result:** Reviewer knows exactly what to focus on and how to debug issues!

## Summary

The heuristic + LLM integration creates a **smart review assistant** that:

1. **Measures** complexity and impact objectively (heuristics)
1. **Understands** the code contextually (LLM)
1. **Guides** reviewers with specific actionable advice (LLM + heuristics)
1. **Adapts** guidance based on PR complexity (simple vs complex)

You get the **best of both worlds**: fast, consistent scoring + nuanced, contextual guidance! 🎉
