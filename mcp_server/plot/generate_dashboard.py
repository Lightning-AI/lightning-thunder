import os
from datetime import datetime
import plotly.graph_objects as go


def generate_comprehensive_dashboard(blocked, not_ready, needs_internal_review, ready_for_external, summary):
    """
    Generate a comprehensive HTML dashboard combining:
    1. Interactive quadrant scatter plot
    2. Beautiful PR cards for each category
    3. Filtering and sorting capabilities
    4. Summary statistics

    Args:
        blocked: List of blocked PRs
        not_ready: List of not ready PRs
        needs_internal_review: List of needs internal review PRs
        ready_for_external: List of ready for external PRs
        summary: Summary of the dashboard

    Returns:
        Path to generated HTML file
    """
    # Combine all PRs with their category labels
    all_prs = []

    for pr in blocked:
        all_prs.append({**pr, "category": "Blocked", "color": "#EF4444"})  # Red

    for pr in not_ready:
        all_prs.append({**pr, "category": "Not Ready", "color": "#F59E0B"})  # Orange

    for pr in needs_internal_review:
        all_prs.append({**pr, "category": "Needs Internal Review", "color": "#3B82F6"})  # Blue

    for pr in ready_for_external:
        all_prs.append({**pr, "category": "Ready for External", "color": "#10B981"})  # Green

    if not all_prs:
        raise ValueError("No PRs to visualize")

    # Create the scatter plot
    fig = go.Figure()

    # Add traces for each category (for legend)
    for category in ["Blocked", "Not Ready", "Needs Internal Review", "Ready for External"]:
        cat_prs = [pr for pr in all_prs if pr["category"] == category]
        if cat_prs:
            fig.add_trace(
                go.Scatter(
                    x=[pr["complexity_score"] for pr in cat_prs],
                    y=[pr["priority_score"] for pr in cat_prs],
                    mode="markers",
                    name=category,
                    marker=dict(size=12, color=cat_prs[0]["color"], line=dict(width=1, color="white"), symbol="circle"),
                    text=[f"PR #{pr['number']}" for pr in cat_prs],
                    hovertext=[
                        f"<b>PR #{pr['number']}: {pr['title'][:50]}</b><br>"
                        f"Author: {pr['author']}<br>"
                        f"Category: {pr['category']}<br>"
                        f"Priority: {pr['priority_score']}/100<br>"
                        f"Complexity: {pr['complexity_score']}/10<br>"
                        f"Impact: {pr['impact_score']}/10<br>"
                        f"Created: {pr['created_at']}<br>"
                        f"<a href='{pr['url']}'>View on GitHub</a>"
                        for pr in cat_prs
                    ],
                    hoverinfo="text",
                    customdata=[pr["url"] for pr in cat_prs],
                )
            )

    # Calculate midpoints for quadrant lines
    mid_x = 5  # Complexity scale is 0-10
    mid_y = 50  # Priority scale is 0-100

    # Add quadrant divider lines
    fig.add_hline(y=mid_y, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=mid_x, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels as annotations
    fig.add_annotation(
        x=2.5,
        y=75,
        text="<b>Q1: Quick Wins</b><br>High Priority<br>Low Complexity<br>🟢 DO FIRST!",
        showarrow=False,
        font=dict(size=11, color="green"),
        bgcolor="rgba(16, 185, 129, 0.1)",
        bordercolor="green",
        borderwidth=1,
        borderpad=8,
    )

    fig.add_annotation(
        x=7.5,
        y=75,
        text="<b>Q2: Major Projects</b><br>High Priority<br>High Complexity<br>🟠 PLAN CAREFULLY",
        showarrow=False,
        font=dict(size=11, color="orange"),
        bgcolor="rgba(245, 158, 11, 0.1)",
        bordercolor="orange",
        borderwidth=1,
        borderpad=8,
    )

    fig.add_annotation(
        x=2.5,
        y=25,
        text="<b>Q3: Easy Backlog</b><br>Low Priority<br>Low Complexity<br>🔵 BACKLOG",
        showarrow=False,
        font=dict(size=11, color="blue"),
        bgcolor="rgba(59, 130, 246, 0.1)",
        bordercolor="blue",
        borderwidth=1,
        borderpad=8,
    )

    fig.add_annotation(
        x=7.5,
        y=25,
        text="<b>Q4: Reconsider</b><br>Low Priority<br>High Complexity<br>🔴 QUESTION VALUE",
        showarrow=False,
        font=dict(size=11, color="red"),
        bgcolor="rgba(239, 68, 68, 0.1)",
        bordercolor="red",
        borderwidth=1,
        borderpad=8,
    )

    # Update layout
    fig.update_layout(
        title={
            "text": f"<b>PR Review Dashboard - Quadrant Analysis</b><br><sup>Total: {summary['total_open_prs']} PRs | "
            f"Ready: {summary['ready_for_external']} | Blocked: {summary['blocked']} | "
            f"Needs Review: {summary['needs_internal_review']} | Not Ready: {summary['not_ready']}</sup>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="<b>Complexity Score</b> (0=Simple, 10=Complex)",
        yaxis_title="<b>Priority Score</b> (0=Low, 100=High)",
        xaxis=dict(range=[-0.5, 10.5], showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="lightgray", zeroline=False),
        hovermode="closest",
        plot_bgcolor="white",
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(title="<b>PR Category</b>", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    # Save plotly chart to HTML string
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": "pr_dashboard", "height": 800, "width": 1200, "scale": 2},
    }

    plotly_html = fig.to_html(include_plotlyjs="cdn", config=config, div_id="plotly-chart")

    # Generate comprehensive HTML with PR cards
    html_content = generate_comprehensive_html(
        plotly_html=plotly_html,
        blocked=blocked,
        not_ready=not_ready,
        needs_internal_review=needs_internal_review,
        ready_for_external=ready_for_external,
        summary=summary,
        all_prs=all_prs,
    )

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "dashboards")
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"comprehensive_dashboard_{timestamp}.html")

    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path


def generate_comprehensive_html(
    plotly_html, blocked, not_ready, needs_internal_review, ready_for_external, summary, all_prs
):
    """Generate the comprehensive HTML content"""

    # Helper function to generate PR card HTML
    def generate_pr_card(pr):
        labels_html = "".join([f'<span class="label-tag">🏷️ {label}</span>' for label in pr.get("labels", [])])

        # Status badge
        if pr.get("category") == "Blocked":
            status_html = '<span class="status-badge status-blocked">🔴 Blocked</span>'
        elif pr.get("category") == "Ready for External":
            status_html = '<span class="status-badge status-ready">✅ Ready for External Review</span>'
        elif pr.get("category") == "Needs Internal Review":
            status_html = '<span class="status-badge status-needs-review">⏳ Needs Internal Review</span>'
        else:
            status_html = '<span class="status-badge status-not-ready">⚠️ Not Ready</span>'

        # Team approvals info
        team_info = pr.get("internal_review", {})
        approvals = team_info.get("team_approvals", 0)
        changes_requested = team_info.get("team_changes_requested", 0)
        reviewers = ", ".join(team_info.get("team_reviewers", []))

        # Definition of ready
        dor = pr.get("definition_of_ready", {})
        readiness_score = dor.get("readiness_score", 0)
        failing_checks = dor.get("failing_checks", [])
        failing_checks_html = "".join([f'<div style="margin-top: 5px;">❌ {check}</div>' for check in failing_checks])

        # Next action
        next_action = pr.get("next_action", pr.get("block_reason", "Review pending"))

        return f'''
        <div class="pr-card" data-category="{pr.get("category", "")}" data-priority="{
            pr.get("priority_score", 0)
        }" data-pr-number="{pr.get("number", 0)}">
            <div class="pr-header">
                <span class="pr-number">#{pr.get("number", "N/A")}</span>
                <h3 class="pr-title">{pr.get("title", "No title")}</h3>
                <div class="pr-meta">
                    <div class="pr-meta-item">👤 <strong>{pr.get("author", "Unknown")}</strong></div>
                    <div class="pr-meta-item">📅 {pr.get("created_at", "Unknown")}</div>
                    <div class="pr-meta-item">🔗 <a href="{pr.get("url", "#")}" target="_blank">View PR</a></div>
                </div>
                <div style="margin-top: 10px;">
                    {status_html}
                </div>
                {'<div class="labels" style="margin-top: 10px;">' + labels_html + "</div>" if labels_html else ""}
            </div>

            <div class="section">
                <div class="section-title">📊 Scores</div>
                <div class="score-grid">
                    <div class="score-item">
                        <div class="score-label">Priority</div>
                        <div class="score-value">{pr.get("priority_score", 0)}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Complexity</div>
                        <div class="score-value">{pr.get("complexity_score", 0)}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Impact</div>
                        <div class="score-value">{pr.get("impact_score", 0)}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Readiness</div>
                        <div class="score-value">{readiness_score}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">👥 Review Status</div>
                <div class="section-content">
                    <div><strong>Team Approvals:</strong> {approvals}/2</div>
                    <div style="margin-top: 5px;"><strong>Changes Requested:</strong> {changes_requested}</div>
                    {
            f'<div style="margin-top: 5px;"><strong>Reviewers:</strong> {reviewers}</div>' if reviewers else ""
        }
                </div>
            </div>

            {
            f"""<div class="section">
                <div class="section-title">📋 Definition of Ready</div>
                <div class="section-content">
                    <div style="color: {"#28a745" if not failing_checks else "#856404"}; font-weight: 600;">
                        {"✅ All checks passed!" if not failing_checks else f"⚠️ {len(failing_checks)} check(s) failing"}
                    </div>
                    {failing_checks_html}
                </div>
            </div>"""
            if failing_checks or readiness_score == 100
            else ""
        }

            <div class="section">
                <div class="section-title">🎯 Next Action</div>
                <div class="section-content" style="font-weight: 600; color: #667eea;">
                    {next_action}
                </div>
            </div>

            <a href="{pr.get("url", "#")}" target="_blank" class="link-button">
                View on GitHub →
            </a>
        </div>
        '''

    # Generate HTML for each category
    blocked_cards = "".join([generate_pr_card(pr) for pr in blocked])
    ready_cards = "".join([generate_pr_card(pr) for pr in ready_for_external])
    needs_review_cards = "".join([generate_pr_card(pr) for pr in needs_internal_review])
    not_ready_cards = "".join([generate_pr_card(pr) for pr in not_ready[:20]])  # Limit to first 20 for performance

    # Full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightning Thunder - Comprehensive PR Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .header h1 {{
            color: #333;
            font-size: 3em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .header .subtitle {{
            color: #666;
            font-size: 1.3em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}

        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            color: #333;
            font-size: 2.8em;
            font-weight: bold;
        }}

        .stat-card .subtext {{
            color: #999;
            font-size: 0.85em;
            margin-top: 8px;
        }}

        .chart-section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .chart-section h2 {{
            color: #333;
            font-size: 2em;
            margin-bottom: 20px;
        }}

        .filters {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filters label {{
            font-weight: 600;
            color: #333;
        }}

        .filters select, .filters input {{
            padding: 8px 15px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }}

        .filters select:focus, .filters input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .category-section {{
            margin-bottom: 50px;
        }}

        .category-header {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .category-header h2 {{
            color: #333;
            font-size: 2.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .category-header .description {{
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }}

        .category-header .count-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.8em;
            font-weight: bold;
        }}

        .pr-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}

        .pr-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .pr-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }}

        .pr-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }}

        .pr-header {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }}

        .pr-number {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 12px;
        }}

        .pr-title {{
            font-size: 1.3em;
            color: #333;
            margin: 12px 0;
            font-weight: 600;
            line-height: 1.4;
        }}

        .pr-meta {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 0.9em;
            color: #666;
            margin-top: 12px;
        }}

        .pr-meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .pr-meta-item a {{
            color: #667eea;
            text-decoration: none;
        }}

        .pr-meta-item a:hover {{
            text-decoration: underline;
        }}

        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 5px 5px 5px 0;
        }}

        .status-ready {{
            background: #d4edda;
            color: #155724;
        }}

        .status-blocked {{
            background: #f8d7da;
            color: #721c24;
        }}

        .status-needs-review {{
            background: #d1ecf1;
            color: #0c5460;
        }}

        .status-not-ready {{
            background: #fff3cd;
            color: #856404;
        }}

        .section {{
            margin: 20px 0;
        }}

        .section-title {{
            font-size: 1.05em;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-content {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            line-height: 1.6;
            font-size: 0.95em;
        }}

        .score-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }}

        .score-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #f0f0f0;
            transition: border-color 0.3s ease;
        }}

        .score-item:hover {{
            border-color: #667eea;
        }}

        .score-label {{
            font-size: 0.75em;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .score-value {{
            font-size: 1.6em;
            font-weight: bold;
            color: #667eea;
        }}

        .labels {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .label-tag {{
            background: #e9ecef;
            color: #495057;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.85em;
        }}

        .link-button {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 15px;
            transition: background 0.3s ease, transform 0.2s ease;
        }}

        .link-button:hover {{
            background: #764ba2;
            transform: scale(1.05);
        }}

        .footer {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            color: #666;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 50px;
        }}

        .footer p {{
            font-size: 1.1em;
        }}

        .nav-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            flex-wrap: wrap;
        }}

        .nav-tab {{
            padding: 12px 24px;
            border-radius: 8px;
            background: #f0f0f0;
            color: #666;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            font-size: 1em;
        }}

        .nav-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .nav-tab:hover {{
            background: #e0e0e0;
        }}

        .nav-tab.active:hover {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}

        @media (max-width: 768px) {{
            .pr-grid {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}

            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>⚡ Lightning Thunder - Comprehensive PR Dashboard</h1>
            <p class="subtitle">Complete overview of all {summary["total_open_prs"]} open pull requests | Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        </div>

        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">📊 Total PRs</div>
                <div class="value">{summary["total_open_prs"]}</div>
                <div class="subtext">Open Pull Requests</div>
            </div>
            <div class="stat-card">
                <div class="label">✅ Ready for External</div>
                <div class="value">{summary["ready_for_external"]}</div>
                <div class="subtext">{summary["pipeline_health"]["ready_percentage"]}% of total</div>
            </div>
            <div class="stat-card">
                <div class="label">⏳ Needs Internal Review</div>
                <div class="value">{summary["needs_internal_review"]}</div>
                <div class="subtext">Waiting for team approval</div>
            </div>
            <div class="stat-card">
                <div class="label">🔴 Blocked</div>
                <div class="value">{summary["blocked"]}</div>
                <div class="subtext">{summary["pipeline_health"]["blocked_percentage"]}% of total</div>
            </div>
            <div class="stat-card">
                <div class="label">⚠️ Not Ready</div>
                <div class="value">{summary["not_ready"]}</div>
                <div class="subtext">Needs author attention</div>
            </div>
            <div class="stat-card">
                <div class="label">🎯 Pipeline Health</div>
                <div class="value">{summary["pipeline_health"]["in_review"]}</div>
                <div class="subtext">In active review</div>
            </div>
        </div>

        <!-- Interactive Quadrant Chart -->
        <div class="chart-section">
            <h2>📈 Priority vs Complexity Quadrant Analysis</h2>
            <p style="color: #666; margin-bottom: 20px;">
                Click on any point to see PR details. Use the legend to filter by category.
            </p>
            {plotly_html}
        </div>

        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showCategory('all')">All PRs ({summary["total_open_prs"]})</button>
            <button class="nav-tab" onclick="showCategory('blocked')">🔴 Blocked ({summary["blocked"]})</button>
            <button class="nav-tab" onclick="showCategory('ready-for-external')">✅ Ready ({summary["ready_for_external"]})</button>
            <button class="nav-tab" onclick="showCategory('needs-internal-review')">⏳ Needs Review ({summary["needs_internal_review"]})</button>
            <button class="nav-tab" onclick="showCategory('not-ready')">⚠️ Not Ready ({summary["not_ready"]})</button>
        </div>

        <!-- Filters -->
        <div class="filters">
            <label>🔍 Search by PR #:</label>
            <input type="text" id="search-pr" placeholder="Enter PR number..." onkeyup="filterPRs()">

            <label>Sort by:</label>
            <select id="sort-by" onchange="sortPRs()">
                <option value="priority">Priority (High to Low)</option>
                <option value="priority-low">Priority (Low to High)</option>
                <option value="number">PR Number (Newest)</option>
                <option value="number-old">PR Number (Oldest)</option>
                <option value="complexity">Complexity (High to Low)</option>
                <option value="complexity-low">Complexity (Low to High)</option>
            </select>
        </div>

        <!-- Blocked PRs Section -->
        <div class="category-section" id="section-blocked">
            <div class="category-header">
                <h2>🔴 Blocked PRs <span class="count-badge">{summary["blocked"]}</span></h2>
                <p class="description">PRs with open change requests from team members. These need immediate author attention.</p>
            </div>
            <div class="pr-grid">
                {blocked_cards if blocked_cards else '<p style="color: white; text-align: center; padding: 40px;">No blocked PRs! 🎉</p>'}
            </div>
        </div>

        <!-- Ready for External Review Section -->
        <div class="category-section" id="section-ready-for-external">
            <div class="category-header">
                <h2>✅ Ready for External Review <span class="count-badge">{summary["ready_for_external"]}</span></h2>
                <p class="description">PRs with 2+ team approvals. Ready to ping external maintainers (@lantiga, @t-vi, @KaelanDt).</p>
            </div>
            <div class="pr-grid">
                {ready_cards if ready_cards else '<p style="color: white; text-align: center; padding: 40px;">No PRs ready for external review yet.</p>'}
            </div>
        </div>

        <!-- Needs Internal Review Section -->
        <div class="category-section" id="section-needs-internal-review">
            <div class="category-header">
                <h2>⏳ Needs Internal Review <span class="count-badge">{summary["needs_internal_review"]}</span></h2>
                <p class="description">PRs that are ready for review but need more team approvals before external review.</p>
            </div>
            <div class="pr-grid">
                {needs_review_cards if needs_review_cards else '<p style="color: white; text-align: center; padding: 40px;">No PRs waiting for internal review.</p>'}
            </div>
        </div>

        <!-- Not Ready Section -->
        <div class="category-section" id="section-not-ready">
            <div class="category-header">
                <h2>⚠️ Not Ready for Review <span class="count-badge">{summary["not_ready"]}</span></h2>
                <p class="description">PRs that fail Definition of Ready checks. Authors need to address failing checks.</p>
            </div>
            <div class="pr-grid">
                {not_ready_cards}
                {f'<p style="color: white; text-align: center; padding: 40px;">Showing first 20 PRs out of {summary["not_ready"]}. Use filters to find specific PRs.</p>' if summary["not_ready"] > 20 else ""}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p style="font-size: 1.3em; font-weight: 600;">⚡ Generated by Lightning Thunder PR Analysis MCP</p>
            <p style="margin-top: 15px; font-size: 1em;">Comprehensive PR Dashboard | {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            <p style="margin-top: 10px; color: #999;">Combining quadrant analysis with detailed PR cards for complete visibility</p>
        </div>
    </div>

    <script>
        // Show/hide categories
        function showCategory(category) {{
            const sections = document.querySelectorAll('.category-section');
            const tabs = document.querySelectorAll('.nav-tab');

            tabs.forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');

            if (category === 'all') {{
                sections.forEach(section => section.style.display = 'block');
            }} else {{
                sections.forEach(section => {{
                    if (section.id === `section-${{category}}`) {{
                        section.style.display = 'block';
                    }} else {{
                        section.style.display = 'none';
                    }}
                }});
            }}
        }}

        // Filter PRs by number
        function filterPRs() {{
            const searchValue = document.getElementById('search-pr').value;
            const cards = document.querySelectorAll('.pr-card');

            cards.forEach(card => {{
                const prNumber = card.getAttribute('data-pr-number');
                if (searchValue === '' || prNumber.includes(searchValue)) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        // Sort PRs
        function sortPRs() {{
            const sortBy = document.getElementById('sort-by').value;
            const grids = document.querySelectorAll('.pr-grid');

            grids.forEach(grid => {{
                const cards = Array.from(grid.querySelectorAll('.pr-card'));

                cards.sort((a, b) => {{
                    if (sortBy === 'priority') {{
                        return parseInt(b.getAttribute('data-priority')) - parseInt(a.getAttribute('data-priority'));
                    }} else if (sortBy === 'priority-low') {{
                        return parseInt(a.getAttribute('data-priority')) - parseInt(b.getAttribute('data-priority'));
                    }} else if (sortBy === 'number') {{
                        return parseInt(b.getAttribute('data-pr-number')) - parseInt(a.getAttribute('data-pr-number'));
                    }} else if (sortBy === 'number-old') {{
                        return parseInt(a.getAttribute('data-pr-number')) - parseInt(b.getAttribute('data-pr-number'));
                    }} else if (sortBy === 'complexity') {{
                        return parseInt(b.getAttribute('data-complexity') || 0) - parseInt(a.getAttribute('data-complexity') || 0);
                    }} else if (sortBy === 'complexity-low') {{
                        return parseInt(a.getAttribute('data-complexity') || 0) - parseInt(b.getAttribute('data-complexity') || 0);
                    }}
                }});

                cards.forEach(card => grid.appendChild(card));
            }});
        }}
    </script>
</body>
</html>"""

    return html


# Legacy function for backward compatibility
def generate_dashboard_html(blocked, not_ready, needs_internal_review, ready_for_external, summary):
    """
    Generate an interactive HTML dashboard with quadrant visualization.

    This is the original function. For the new comprehensive dashboard, use generate_comprehensive_dashboard().
    """
    return generate_comprehensive_dashboard(blocked, not_ready, needs_internal_review, ready_for_external, summary)


def generate_dashboard_recommendations(summary, ready_for_external, needs_internal_review, blocked, not_ready):
    """Generate actionable recommendations based on dashboard state.

    Args:
        summary: Summary of the dashboard
        ready_for_external: List of ready for external PRs
        needs_internal_review: List of needs internal review PRs
        blocked: List of blocked PRs
        not_ready: List of not ready PRs

    Returns:
        List of recommendations
    """
    recommendations = []

    # Critical: Blocked PRs
    if blocked:
        top_blocked = blocked[:3]
        recommendations.append(
            {
                "priority": "🔴 CRITICAL",
                "action": f"Unblock {len(blocked)} PR(s) with change requests",
                "prs": [f"#{pr['number']}: {pr['title'][:50]}" for pr in top_blocked],
                "reason": "These PRs are blocked and need author attention",
                "next_steps": "Authors should address feedback and request re-review",
            }
        )

    # High Priority: Ready for external review
    if ready_for_external:
        top_ready = ready_for_external[:5]
        recommendations.append(
            {
                "priority": "🟠 HIGH",
                "action": f"Ping external maintainers for {len(ready_for_external)} PR(s)",
                "prs": [f"#{pr['number']}: {pr['title'][:50]}" for pr in top_ready],
                "reason": "These PRs have 2+ team approvals and are ready for external review",
                "next_steps": "Request review from @lantiga, @t-vi, or @KaelanDt",
            }
        )

    # Medium: Not ready PRs
    if not_ready:
        top_not_ready = not_ready[:3]
        recommendations.append(
            {
                "priority": "🟡 MEDIUM",
                "action": f"Fix {len(not_ready)} PR(s) failing Definition of Ready",
                "prs": [f"#{pr['number']}: {pr['title'][:50]} - {pr['not_ready_reason'][:50]}" for pr in top_not_ready],
                "reason": "These PRs need author action before they can be reviewed",
                "next_steps": "Authors should address failing checks",
            }
        )

    # Medium: Needs internal review
    if needs_internal_review:
        high_priority_review = [pr for pr in needs_internal_review if pr["priority_score"] >= 70]
        if high_priority_review:
            top_review = high_priority_review[:5]
            recommendations.append(
                {
                    "priority": "🟡 MEDIUM",
                    "action": f"Review {len(high_priority_review)} high-priority PR(s)",
                    "prs": [
                        f"#{pr['number']}: {pr['title'][:50]} (priority: {pr['priority_score']})" for pr in top_review
                    ],
                    "reason": "High-priority PRs waiting for internal review",
                    "next_steps": "Team members should review and approve",
                }
            )

    # Overall health assessments
    if summary["pipeline_health"]["blocked_percentage"] > 30:
        recommendations.append(
            {
                "priority": "🔴 ALERT",
                "action": "Pipeline bottleneck: High blocked rate",
                "reason": f"{summary['pipeline_health']['blocked_percentage']}% of PRs are blocked",
                "next_steps": "Consider a focused sprint to clear change requests",
            }
        )

    if summary["pipeline_health"]["ready_to_merge"] > 10:
        recommendations.append(
            {
                "priority": "🟠 ALERT",
                "action": "External review bottleneck detected",
                "reason": f"{summary['pipeline_health']['ready_to_merge']} PRs are ready but waiting for external maintainers",
                "next_steps": "Batch ping maintainers or schedule dedicated review time",
            }
        )

    if summary["pipeline_health"]["needs_attention"] > summary["pipeline_health"]["in_review"] * 2:
        recommendations.append(
            {
                "priority": "🟡 INFO",
                "action": "Many PRs need author attention",
                "reason": f"{summary['pipeline_health']['needs_attention']} PRs are blocked or not ready",
                "next_steps": "Consider a PR cleanup sprint or author check-in",
            }
        )

    return recommendations
