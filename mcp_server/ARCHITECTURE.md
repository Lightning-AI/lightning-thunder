# Architecture: Google Drive RAG Integration

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CURSOR IDE                                    │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        User (Developer)                             │   │
│  │                                                                     │   │
│  │  Commands:                                                          │   │
│  │  • analyze_single_pr(123)                                           │   │
│  │  • llm_batch_analysis(min_priority=30)                              │   │
│  │  • mcp_MaaS_Google_Drive_gdrive_search("coding standards")          │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                             │
│                              │ MCP Protocol                                │
│                              ▼                                             │
│  ┌────────────────────────────────────────────┐  ┌──────────────────────┐  │
│  │   Thunder PR Inspector MCP Server          │  │  MaaS Google Drive   │  │
│  │                                            │  │  MCP Server          │  │
│  │                                            │  │                      │  │
│  │  ┌──────────────────────────────────────┐  │  │  ┌────────────────┐  │  │
│  │  │  PR Analysis Engine                  │  │  │  │  Search        │  │  │
│  │  │  • GitHub API calls                  │  │  │  │  Get Files     │  │  │
│  │  │  • Heuristic scoring                 │  │  │  │  Get Metadata  │  │  │
│  │  │  • Risk assessment                   │  │  │  │  Health Check  │  │  │
│  │  └──────────────────────────────────────┘  │  │  └────────────────┘  │  │
│  │                                            │  │          │           │  │
│  │  ┌──────────────────────────────────────┐  │  │          │           │  │
│  │  │  GoogleDriveContextManager (RAG)     │  │  │          │           │  │
│  │  │  ┌────────────────────────────────┐  │  │  │          │           │  │
│  │  │  │ 1. Analyze PR characteristics  │  │  │  │          │           │  │
│  │  │  │    • Title keywords            │  │  │  │          │           │  │
│  │  │  │    • Labels                    │  │  │  │          │           │  │
│  │  │  │    • Description               │  │  │  │          │           │  │
│  │  │  └────────────────────────────────┘  │  │  │          │           │  │
│  │  │           │                          │  │  │          │           │  │
│  │  │           ▼                          │  │  │          │           │  │
│  │  │  ┌────────────────────────────────┐  │  │  │          │           │  │
│  │  │  │ 2. Build search queries        │  │  │  │          │           │  │
│  │  │  │    • "bug fixing standards"    │──┼─ ┼──┼──────────┘           │  │
│  │  │  │    • "testing guidelines"      │  │  │                         │  │
│  │  │  │    • "architecture overview"   │  │  │                         │  │
│  │  │  └────────────────────────────────┘  │  │          ▲              │  │
│  │  │           │                          │  │          │              │  │
│  │  │           ▼                          │  │          │ Search API   │  │
│  │  │  ┌────────────────────────────────┐  │  │          │              |  │
│  │  │  │ 3. Fetch documents             │  │  │          │              │  │
│  │  │  │    Via MaaS MCP ───────────────┼──┼─ ┼──────────┘              │  │
│  │  │  └────────────────────────────────┘  │  │                         │  │
│  │  │           │                          │  │                         │  │
│  │  │           ▼                          │  │                         │  │
│  │  │  ┌────────────────────────────────┐  │  │                         │  │
│  │  │  │ 4. Cache & format context      │  │  │                         │  │
│  │  │  └────────────────────────────────┘  │  │                         │  │
│  │  └──────────────────┬───────────────────┘  │                         │  │
│  │                     │                      │                         │  │
│  │                     ▼                      │                         │  │
│  │  ┌──────────────────────────────────────┐  │                         │  │
│  │  │  LLM Prompt Builder                  │  │                         │  │
│  │  │  • PR data + heuristics              │  │                         │  │
│  │  │  • Google Drive context (RAG)        │  │                         │  │
│  │  │  • Comprehensive analysis prompt     │  │                         │  │
│  │  └──────────────────────────────────────┘  │                         │  │
│  └────────────────────────────────────────────┘                            │
│                               │                                            │
│                               │ Enhanced Prompt                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Claude LLM (Cursor)                             │   │
│  │                                                                     │   │
│  │  Input: PR data + GitHub info + Google Drive documentation          │   │
│  │  Output: Enhanced analysis with organizational context              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                            │
│                               │ Analysis Result                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          User Receives                              │   │
│  │  • PR summary                                                       │   │
│  │  • Risk assessment (informed by organizational standards)           │   │
│  │  • Priority score                                                   │   │
│  │  • Recommendations (aligned with company guidelines)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

```

## Data Flow: Single PR Analysis

Review a single PR using both heuristic and LLM methods

```
analyze_single_pr(123)
    │
    ├─► 1. Fetch PR from GitHub
    │       • PR metadata (title, labels, author, dates)
    │       • PR diff (code changes)
    │       • Reviews, comments, files changed
    │
    ├─► 2. Run Heuristic Analysis
    │       • Calculate risk scores
    │       • Assess staleness
    │       • Review status
    │       • Priority score
    │
    ├─► 3. Build Context (kind of RAG)
    │       │
    │       ├─► Analyze PR characteristics
    │       │   • Title: "Fix authentication bug in JWT validation"
    │       │   • Labels: ["bug", "security"]
    │       │   • → Queries: ["bug fixing standards", "security practices"]
    │       │
    │       ├─► Search Google Drive Cache
    │       │   • Check for cached documents
    │       │
    │       └─► Format Context
    │           • Combine documents
    │           • Add section headers
    │           • Truncate if needed
    │
    ├─► 4. Build Enhanced LLM Prompt
    │       ┌────────────────────────────────────────┐
    │       │ You are a Senior Staff Engineer...     │
    │       │                                        │
    │       │ PR Title: Fix authentication bug...    │
    │       │ PR Body: ...                           │
    │       │ Code Diff: ...                         │
    │       │                                        │
    │       │ ## ADDITIONAL CONTEXT (NEW!)           │
    │       │                                        │
    │       │ ### Security Best Practices v2.0       │
    │       │ [Content from Google Drive doc 1...]   │
    │       │                                        │
    │       │ ### Bug Fix Guidelines                 │
    │       │ [Content from Google Drive doc 2...]   │
    │       │                                        │
    │       │ Please consider the above when...      │
    │       │                                        │
    │       │ Provide: Summary & Risk Assessment     │
    │       └────────────────────────────────────────┘
    │
    ├─► 5. Execute LLM Analysis
    │       • Send prompt to LLM (via Cursor)
    │       • LLM uses both PR data AND organizational context
    │       • Returns enhanced analysis
    │
    └─► 6. Return Result
            {
              "number": 123,
              "title": "Fix authentication bug...",
              "summary": "...",
              "risk_score": {...},
              "llm_summary": "... (informed by security best practices)",
              "llm_risk_assessment": "... (aligned with org guidelines)",
              ...
            }
```

## Data Flow: Batch Analysis

Analyse all the open PRs, or a `limit` of those, with heuristic + LLM

```
llm_batch_analysis(min_priority=30, limit=10)
    │
    ├─► 1. Fetch All Open PRs from GitHub
    │       • 50+ open PRs retrieved
    │
    ├─► 2. Run Heuristic Analysis on Each
    │       • Calculate priority scores
    │       • Filter: keep only priority >= 30
    │       • Sort by priority (highest first)
    │       • Limit to 10 PRs
    │
    ├─► 3. Build Context (kind of RAG)
    │       │
    │       ├─► Search for cached documents
    │
    ├─► 4. Build Comprehensive Batch Prompt
    │       ┌────────────────────────────────────────┐
    │       │ You are a Senior Staff Engineer...     │
    │       │                                        │
    │       │ ## PROJECT CONTEXT (NEW!)              │
    │       │ [Architecture overview from GDrive...] │
    │       │ [Design principles from GDrive...]     │
    │       │                                        │
    │       │ ## PULL REQUESTS TO ANALYZE            │
    │       │                                        │
    │       │ ### PR #123: Fix authentication bug    │
    │       │ Heuristic Priority: 85/100             │
    │       │ Risk: Breaking=2, Security=8, ...      │
    │       │ [PR details...]                        │
    │       │                                        │
    │       │ ### PR #456: Add caching layer         │
    │       │ Heuristic Priority: 72/100             │
    │       │ [PR details...]                        │
    │       │                                        │
    │       │ [... 8 more PRs ...]                   │
    │       │                                        │
    │       │ YOUR TASK:                             │
    │       │ 1. Score each PR (0-100)               │
    │       │ 2. Prioritize review order             │
    │       │ 3. Identify critical/safe to merge     │
    │       │ 4. Overall assessment                  │
    │       └────────────────────────────────────────┘
    │
    └─► 5. Return Prompt + Data
            {
              "total_prs_analyzed": 10,
              "llm_analysis_prompt": "... (with project context)",
              "heuristic_data": [...]
            }
```
