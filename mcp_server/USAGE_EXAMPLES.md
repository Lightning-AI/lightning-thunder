# Usage Examples: Explicit Google Drive Files

## Quick Reference

### Without Google Drive Files

```python
# Basic PR analysis
analyze_single_pr(pr_number=123)
```

### With Google Drive Files

```python
# PR analysis calibrated with specific documents
analyze_single_pr(pr_number=123, gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"])
```

______________________________________________________________________

## Real Examples

### Example 1: Quarterly Prioritization

**Scenario:** You want to prioritize PRs based on Q4 objectives.

```python
llm_batch_analysis(min_priority=30, limit=15, gdrive_files=["ThunderQ4Plan"])
```

**Output:** All PRs analyzed and prioritized according to Q4 goals documented in "ThunderQ4Plan".

______________________________________________________________________

### Example 2: Security Review

**Scenario:** Security-critical PR needs review against company guidelines.

```python
analyze_single_pr(pr_number=456, gdrive_files=["SecurityGuidelines", "ThreatModel"])
```

**Output:** Analysis calibrated with security standards from the specified documents.

______________________________________________________________________

### Example 3: Architecture Decision

**Scenario:** Major architectural change needs validation against design principles.

```python
analyze_single_pr(
    pr_number=789,
    gdrive_files=["ThunderArchitecture", "DesignPrinciples", "APIGuidelines"],
)
```

**Output:** Assessment aligned with architectural standards and design principles.

______________________________________________________________________

### Example 4: Multiple PRs, Same Standards

**Scenario:** Review 10 PRs consistently using same reference documents.

```python
# Define your reference files once
ref_files = ["ThunderBestPractices", "CodingStandards"]

# Analyze multiple PRs with consistent context
for pr_num in [101, 102, 103, 104, 105]:
    result = analyze_single_pr(pr_number=pr_num, gdrive_files=ref_files)
    # All PRs analyzed with same standards
```

**Output:** Consistent analysis across all PRs using the same reference documentation.

______________________________________________________________________

### Example 5: Using Full URLs

**Scenario:** You have the exact Google Drive URLs for your documents.

```python
analyze_single_pr(
    pr_number=234,
    gdrive_files=[
        "https://drive.google.com/file/d/ABC123_Q4Plan/view",
        "https://drive.google.com/file/d/XYZ789_BestPractices/view",
    ],
)
```

**Output:** Same as using file names, but uses exact URLs.

______________________________________________________________________

### Example 6: Conversational Request

**Scenario:** You ask the AI agent in natural language.

**You say:**

> "Hey agent, can you analyze PR #345 and calibrate your findings based on Thunder Q4 plan (file on gdrive called: ThunderQ4Plan) and Thunder Best practices (file on gdrive called: ThunderBestPractices)?"

**Agent executes:**

```python
analyze_single_pr(pr_number=345, gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"])
```

**Agent responds:**

> "I've analyzed PR #345 using ThunderQ4Plan and ThunderBestPractices as reference. Here's the assessment: [detailed analysis aligned with Q4 goals and best practices]..."

______________________________________________________________________

### Example 7: No Files Specified

**Scenario:** You want basic analysis without any Google Drive context.

```python
# Just analyze the PR based on code and GitHub data
analyze_single_pr(pr_number=567)
```

**Output:** Standard analysis without additional context (faster, fewer tokens).

______________________________________________________________________

### Example 8: Batch with Different Filters

**Scenario:** Analyze only bug-related PRs with bug-fixing guidelines.

```python
llm_batch_analysis(
    min_priority=20,
    labels=["bug"],
    limit=10,
    gdrive_files=["BugFixGuidelines", "TestingStandards"],
)
```

**Output:** Bug PRs prioritized according to bug-fixing and testing standards.

______________________________________________________________________

### Example 9: Performance Focus

**Scenario:** Weekly performance PR review.

```python
# Get all PRs with performance label
llm_batch_analysis(
    labels=["performance"],
    limit=20,
    gdrive_files=["PerformanceGoals", "OptimizationGuide"],
)
```

**Output:** Performance PRs prioritized based on performance goals and optimization guidelines.

______________________________________________________________________

### Example 10: Team Onboarding

**Scenario:** Review new team member's first PRs with onboarding materials.

```python
new_member_prs = [111, 222, 333]
onboarding_files = ["OnboardingGuide", "CodingStandards", "TeamConventions"]

for pr_num in new_member_prs:
    analyze_single_pr(pr_number=pr_num, gdrive_files=onboarding_files)
```

**Output:** Helpful feedback aligned with onboarding materials and team conventions.

______________________________________________________________________

## What Gets Added to the Prompt

When you specify `gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"]`, the system:

1. **Searches** for each file in Google Drive
1. **Fetches** the content (up to 10,000 characters per file)
1. **Adds** to the LLM prompt like this:

```
## REFERENCE DOCUMENTATION
The following documentation should be used to calibrate this analysis:

### Lightning Thunder Q4 2024 Plan
[Full content of ThunderQ4Plan...]

### Lightning Thunder Best Practices
[Full content of ThunderBestPractices...]

Please use the above documentation to calibrate your assessment of this PR.
Your analysis should align with the goals, standards, and priorities outlined in these documents.
```

4. **Analyzes** the PR with this additional context

______________________________________________________________________

## Comparison

### Without Files

```python
analyze_single_pr(123)
```

**LLM sees:**

- PR title, description, diff
- GitHub metadata
- Heuristic scores

**Analysis based on:** General software engineering best practices

### With Files

```python
analyze_single_pr(123, gdrive_files=["ThunderQ4Plan"])
```

**LLM sees:**

- PR title, description, diff
- GitHub metadata
- Heuristic scores
- **Your Q4 plan content**

**Analysis based on:** Your specific Q4 goals and priorities

______________________________________________________________________

## Tips

✅ **Use 1-3 files** for best results (avoid token limits)
✅ **Clear names** like "ThunderQ4Plan" work better than "Doc1"
✅ **Keep documents updated** in Google Drive
✅ **Consistent standards** across similar PR reviews
✅ **Leave empty** (`gdrive_files=None`) when you don't need context

______________________________________________________________________

## Getting Started

1. **Organize your Google Drive:**

   - Create/rename key documents with clear names
   - e.g., "ThunderQ4Plan", "ThunderBestPractices"

1. **Test with one PR:**

   ```python
   analyze_single_pr(pr_number=123, gdrive_files=["ThunderBestPractices"])
   ```

1. **Check the logs** (stderr) for:

   ```
   🔍 Searching for file: ThunderBestPractices
   ✓ Added: Lightning Thunder Best Practices (8432 chars)
   ✓ Added Google Drive context to PR #123 analysis (1 files)
   ```

1. **Review the analysis** - it should align with your best practices!

______________________________________________________________________

## That's It!

Simple, explicit, and fully under your control. 🎉
