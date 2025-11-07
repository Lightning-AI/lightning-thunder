# Google Drive Integration - Simple Guide

## What You Asked For ✅

You can now tell the system **exactly which Google Drive files to use** for PR analysis. No automatic searching - full control.

## How It Works

### Basic Command

```python
analyze_single_pr(pr_number=123, gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"])
```

**What happens:**

1. Finds files named "ThunderQ4Plan" and "ThunderBestPractices" in your Google Drive
1. Fetches their content
1. Adds them to the LLM prompt as "REFERENCE DOCUMENTATION"
1. LLM analyzes the PR using these documents to calibrate its assessment

### Your Example

**You say:**

> "Hey agent, can you analyze PR #345 and calibrate your findings based on Thunder Q4 plan (file on gdrive called: ThunderQ4Plan) and Thunder Best practices (file on gdrive called: ThunderBestPractices)?"

**Agent does:**

```python
analyze_single_pr(pr_number=345, gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"])
```

**Agent responds with:** Analysis aligned with your Q4 plan and best practices! ✨

## Batch Analysis

Same thing for multiple PRs:

```python
llm_batch_analysis(
    min_priority=30, limit=10, gdrive_files=["ThunderQ4Plan", "ThunderBestPractices"]
)
```

All 10 PRs are analyzed using the same reference documents.

## Without Google Drive Files

If you don't specify files, it works as before:

```python
# No Google Drive context
analyze_single_pr(pr_number=123)
```

## File Names vs URLs

You can use either:

```python
# Option 1: File names (simpler)
gdrive_files = ["ThunderQ4Plan", "ThunderBestPractices"]

# Option 2: Full URLs (more explicit)
gdrive_files = [
    "https://drive.google.com/file/d/ABC123/view",
    "https://drive.google.com/file/d/XYZ789/view",
]
```

## What You Get

### Without Files

```
PR Analysis = GitHub Data + Heuristics + LLM
```

### With Files

```
PR Analysis = GitHub Data + Heuristics +
              Your Specific Documents + LLM

→ Analysis aligned with YOUR standards and goals
```

## Setup

1. **Organize your Google Drive**

   - Name files clearly: "ThunderQ4Plan", "ThunderBestPractices", etc.

1. **Test it**

   ```python
   analyze_single_pr(pr_number=123, gdrive_files=["ThunderBestPractices"])
   ```

1. **Check stderr logs**

   ```
   🔍 Searching for file: ThunderBestPractices
   ✓ Added: Lightning Thunder Best Practices (8432 chars)
   ✓ Added Google Drive context to PR #123 analysis (1 files)
   ```

1. **Done!** 🎉

## Documentation

- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Real-world examples
- **[EXPLICIT_FILES_GUIDE.md](EXPLICIT_FILES_GUIDE.md)** - Complete guide
- **[server.py](server.py)** - The code

## Key Features

✅ **Explicit control** - You specify which files
✅ **No automatic searching** - Only uses files you list
✅ **Simple API** - Just pass file names
✅ **Works with MaaS Google Drive MCP** - Uses your organization's MCP
✅ **Optional** - Leave empty to skip Google Drive

## Summary

```python
# Your workflow is now:
analyze_single_pr(pr_number=YOUR_PR, gdrive_files=["YOUR_FILE_1", "YOUR_FILE_2"])

# That's it! 🚀
```

The system finds those specific files, fetches them, and uses them to calibrate the PR analysis. Exactly what you asked for! 💯
