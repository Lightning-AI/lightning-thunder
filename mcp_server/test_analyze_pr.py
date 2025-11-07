#!/usr/bin/env python3
"""
Test script to run analyze_pr directly without MCP server
"""
import os
import httpx
import sys
import json

# Add mcp_server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_server'))

from llm_analysis.engine import analyze_pr
from utils.helper_functions import dataclass_to_dict

# Make sure GITHUB_TOKEN is set
if not os.getenv("GITHUB_TOKEN"):
    print("ERROR: GITHUB_TOKEN environment variable not set")
    sys.exit(1)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "Lightning-AI"
REPO_NAME = "lightning-thunder"
BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Create the github client (single instance, reused across all calls)
github_client = httpx.Client(base_url=BASE_URL, headers=HEADERS)

# Test analyze_pr with PR #2721
pr_number = 2721

# Option 1: Without Google Drive context
print(f"Analyzing PR #{pr_number} without Google Drive context...")
try:
    analysis = analyze_pr(pr_number)
    print("\n" + "="*80)
    print("ANALYSIS RESULT:")
    print("="*80)
    print(json.dumps(dataclass_to_dict(analysis), indent=2))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")

# Option 2: With Google Drive context
print(f"Analyzing PR #{pr_number} WITH Google Drive context...")
gdrive_files = [
    "Thunder Q4 2025 Plan: Inference",
    "Thunder Team: Pull Request & Code Review Guidelines"
]
try:
    analysis = analyze_pr(pr_number, gdrive_files=gdrive_files, github_client=github_client)
    print("\n" + "="*80)
    print("ANALYSIS RESULT (with GDrive context):")
    print("="*80)
    print(json.dumps(dataclass_to_dict(analysis), indent=2))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
