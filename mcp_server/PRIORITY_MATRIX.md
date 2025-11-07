# PR Priority Matrix

## Overview

The priority system now uses a **Complexity × Impact × Staleness** matrix to intelligently prioritize PRs based on effort required vs. value delivered.

## Priority Matrix

```
┌─────────────────────────────────────────────────────────────┐
│                    PRIORITY MATRIX                          │
│                                                             │
│  Impact                                                     │
│    ▲                                                        │
│  10│  🎯 IMPORTANT          🔥 CRITICAL                      
│    │  Complex + High        Simple + High                   │
│    │  (65-79)               (90-100)                        │
│    │  Needs careful review  Review ASAP!                    │
│  7 ├──────────────────────────────────────                  │
│    │                                                        │
│    │  📝 LOW                ⚡ QUICK WIN                     │
│    │  Complex + Low         Simple + Low                    |
│    │  (0-59)                (70-89)                         │
│  0 │  Deprioritize          Easy wins, do quickly           │
│    └──────────────────────────────────────────► Complexity  │
│         0           4                    10                 │
│      (Simple)   (Moderate)          (Complex)               │
└─────────────────────────────────────────────────────────────┘
```

## The Four Quadrants

### 🔥 CRITICAL (90-100): Simple + High Impact
**Definition:** Easy to review, high value  
**Examples:**
- Security fix with small code change
- Critical bug fix (3 files changed)
- Documentation fix for major feature
- Approved PR ready to merge (unblocking others)

**Action:** Review immediately! These are your highest ROI.

**Staleness Boost:** +15 points if >90 days old

---

### ⚡ QUICK WIN (70-89): Simple + Low Impact  
**Definition:** Easy to review, lower value  
**Examples:**
- Typo fixes
- Minor documentation updates
- Code style improvements
- Small refactoring (non-critical)

**Action:** Review quickly to clear the queue

**Staleness Boost:** +10 points if >60 days old  
**Rationale:** These are "easy wins" - get them done to keep velocity high

---

### 🎯 IMPORTANT (60-79): Complex + High Impact
**Definition:** Requires careful review, high value  
**Examples:**
- Major feature implementation
- Architecture changes
- Performance optimization (large changes)
- Security improvements (complex)

**Action:** Schedule dedicated time, possibly get help from LLM to understand

**Staleness Boost:** +5 points if >90 days old  
**Rationale:** High value but needs thorough review

---

### 📝 LOW (0-59): Complex + Low Impact
**Definition:** High effort, low value  
**Examples:**
- Large refactoring (non-critical)
- Experimental features
- Over-engineered solutions
- Complex changes to non-critical code

**Action:** Deprioritize or request simplification

**No staleness boost** (focus elsewhere first)

---

## Complexity Assessment (0-10)

### Simple (0-3)
- Few files changed (< 5)
- Small line count (< 100 lines)
- Keywords: "fix typo", "update doc", "formatting", "style"
- Non-core files

### Moderate (4-6)
- Medium files changed (5-10)
- Medium line count (100-500 lines)
- Standard feature additions
- Some core file changes

### Complex (7-10)
- Many files changed (> 10)
- Large line count (> 500 lines)
- Keywords: "refactor", "architecture", "redesign", "migration"
- Multiple core file changes

---

## Impact Assessment (0-10)

### Low Impact (0-3)
- Documentation changes
- Style/formatting
- Non-critical improvements
- Labels: "documentation", "chore", "style"

### Medium Impact (4-6)
- Feature additions
- Standard improvements
- Moderate urgency
- Standard risk levels

### High Impact (7-10)
- Security issues (risk ≥ 7)
- Critical bugs
- Breaking changes
- Approved PRs ready to merge (unblocking work)
- Labels: "critical", "blocker", "security", "performance"

---

## Staleness Adjustments

### Simple PRs (Complexity ≤ 4)
**Philosophy:** Get them done! They're easy and sitting too long.

| Age | Adjustment | Reasoning |
|-----|-----------|-----------|
| > 90 days | +15 | "Stale simple PR - knock it out!" |
| > 60 days | +10 | "Aging simple PR - do it soon" |
| > 30 days | +5 | "Waiting simple PR" |

### Complex PRs (Complexity > 4)
**Philosophy:** Smaller boost - these take time regardless.

| Age | Adjustment | Reasoning |
|-----|-----------|-----------|
| > 90 days | +5 | "Stale - check if still relevant" |
| > 60 days | +3 | "Aging" |
| > 30 days | 0 | "Normal for complex PRs" |

### Penalties

| Condition | Adjustment | Reasoning |
|-----------|-----------|-----------|
| Has conflicts | -20 | Needs work before review |
| Changes requested | -10 | Author needs to address feedback |
| No activity + complex | -10 | Likely abandoned or blocked |

### Bonuses

| Condition | Adjustment | Reasoning |
|-----------|-----------|-----------|
| Approved + mergeable | +15 | Ready to go, unblock team! |

---

## Real-World Examples

### Example 1: Typo Fix (120 days old)
```
Complexity: 1 (1 file, 2 lines, "fix typo")
Impact: 2 (documentation label)
Base Score: 75 (⚡ QUICK WIN)
Staleness: +15 (stale simple PR)
Final: 90/100 🔥 CRITICAL

Reasoning: "Easy fix that's been waiting way too long - just do it!"
```

### Example 2: Security Bug Fix (5 files)
```
Complexity: 3 (5 files, 50 lines)
Impact: 9 (security risk = 8)
Base Score: 90 (🔥 CRITICAL)
Staleness: +0 (only 5 days old)
Final: 90/100 🔥 CRITICAL

Reasoning: "Simple + high security impact - immediate attention needed"
```

### Example 3: Major Refactoring (200 days old)
```
Complexity: 8 (25 files, 2000 lines, "refactor")
Impact: 4 (no critical labels)
Base Score: 40 (📝 LOW)
Staleness: +5 (stale but complex)
Final: 45/100 📝 LOW

Reasoning: "Large effort, low urgency - deprioritize"
```

### Example 4: Performance Optimization (Approved)
```
Complexity: 6 (12 files, 300 lines)
Impact: 8 (performance label, approved)
Base Score: 65 (🎯 IMPORTANT)
Staleness: +15 (approved + mergeable)
Final: 80/100 🔥 

Reasoning: "Complex but high impact, approved and ready - merge it!"
```

### Example 5: Large Refactor with Conflicts
```
Complexity: 9 (30 files, 3000 lines)
Impact: 3 (refactor label)
Base Score: 40 (📝 LOW)
Staleness: -20 (has conflicts)
Final: 20/100 📝 LOW

Reasoning: "Complex, low impact, and has conflicts - author needs to fix first"
```

---

## Using the Priority System

### Daily Workflow

1. **Run prioritization:**
```python
prioritize_prs(min_priority=70)
```

2. **Review by category:**
   - **90-100 (🔥 CRITICAL):** Review immediately
   - **70-89 (⚡ QUICK WIN):** Batch review (30 min)
   - **60-79 (🎯 IMPORTANT):** Schedule dedicated time
   - **0-59 (📝 LOW):** Review only if time permits

### Weekly Triage

```python
# Focus on quick wins and critical items
llm_batch_analysis(
    min_priority=70,
    limit=20,
    gdrive_files=["ThunderQ4Plan"]
)
```

### Monthly Cleanup

```python
# Find stale simple PRs
prioritize_prs(min_priority=0)
# Look for simple PRs with high staleness adjustments
# Close or merge old simple PRs
```

---

## Calibrating with Google Drive Files

The impact assessment can be enhanced with your organizational documents:

```python
analyze_single_pr(
    pr_number=123,
    gdrive_files=["ThunderQ4Plan", "ThunderPriorities"]
)
```

The LLM can then assess:
- Does this PR align with Q4 goals? (increase impact)
- Is this in the current sprint? (increase impact)
- Is this technical debt cleanup? (may decrease impact)

---

## Benefits of This System

✅ **Clears stale simple PRs** - Don't let easy wins rot  
✅ **Focuses effort wisely** - Hard work on high-value items  
✅ **Prevents bikeshedding** - Don't spend hours on low-impact complex PRs  
✅ **Unblocks team** - Approved PRs get merged quickly  
✅ **Data-driven** - Clear reasoning for every priority  

---

## Adjusting the Matrix

You can tune the thresholds in `server.py`:

```python
# Adjust complexity threshold
is_simple = complexity <= 4  # Current: 4, increase for stricter

# Adjust impact threshold
is_high_impact = impact >= 7  # Current: 7, decrease for more critical PRs

# Adjust base scores
base_score = 90  # CRITICAL
base_score = 75  # QUICK WIN
base_score = 65  # IMPORTANT
base_score = 40  # LOW

# Adjust staleness boosts
staleness_adjustment += 15  # Simple + very stale
staleness_adjustment += 10  # Simple + stale
```

---

## Summary

**Simple rule of thumb:**
- **Easy + Important** → Do now
- **Easy + Not important** → Do soon (quick wins)
- **Hard + Important** → Schedule carefully
- **Hard + Not important** → Question if it's needed

**Staleness amplifier:**
- Simple PRs get big staleness boost (get them done!)
- Complex PRs get small staleness boost (they take time anyway)

This creates a virtuous cycle: quick wins get done quickly, team velocity stays high, important complex work gets proper attention.
