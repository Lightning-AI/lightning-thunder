import os
from datetime import datetime
import plotly.graph_objects as go


def generate_dashboard_html(blocked, not_ready, needs_internal_review, ready_for_external, summary):
    """
    Generate an interactive HTML dashboard with quadrant visualization.

    Quadrants:
    - Q1 (High Priority, Low Complexity): Quick Wins - Do First!
    - Q2 (High Priority, High Complexity): Major Projects - Plan Carefully
    - Q3 (Low Priority, Low Complexity): Easy Backlog
    - Q4 (Low Priority, High Complexity): Reconsider Value

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
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(title="<b>PR Category</b>", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "dashboards")
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"pr_dashboard_{timestamp}.html")

    # Save the figure with custom JavaScript for sticky tooltips
    # This allows users to hover over the tooltip and click links
    custom_js = """
    <script>
    // Make tooltips sticky and clickable
    (function() {
        let hoverTimeout = null;
        let currentHoverElement = null;

        // Wait for plotly to load
        setTimeout(function() {
            const plotDiv = document.querySelector('.plotly');
            if (!plotDiv) return;

            // Add click handler to open GitHub PR
            plotDiv.on('plotly_click', function(data) {
                if (data.points && data.points.length > 0) {
                    const point = data.points[0];
                    const hovertext = point.hovertext || point.text;
                    // Extract URL from hovertext
                    const urlMatch = hovertext.match(/href='([^']+)'/);
                    if (urlMatch && urlMatch[1]) {
                        window.open(urlMatch[1], '_blank');
                    }
                }
            });

            // Make hover labels clickable and sticky
            plotDiv.on('plotly_hover', function(data) {
                clearTimeout(hoverTimeout);

                // Add event listeners to hover labels to keep them visible
                setTimeout(function() {
                    const hoverLabels = document.querySelectorAll('.hoverlayer .hovertext');
                    hoverLabels.forEach(function(label) {
                        if (label !== currentHoverElement) {
                            currentHoverElement = label;

                            // Make links clickable
                            const links = label.querySelectorAll('a');
                            links.forEach(function(link) {
                                link.style.pointerEvents = 'auto';
                                link.style.cursor = 'pointer';
                                link.style.color = '#3B82F6';
                                link.style.textDecoration = 'underline';
                            });

                            // Keep tooltip visible when hovering over it
                            label.addEventListener('mouseenter', function() {
                                clearTimeout(hoverTimeout);
                            });

                            label.addEventListener('mouseleave', function() {
                                hoverTimeout = setTimeout(function() {
                                    Plotly.Fx.unhover(plotDiv);
                                }, 500);
                            });
                        }
                    });
                }, 50);
            });

            // Add delay before hiding tooltip when leaving a point
            plotDiv.on('plotly_unhover', function(data) {
                hoverTimeout = setTimeout(function() {
                    const hoverLabels = document.querySelectorAll('.hoverlayer .hovertext');
                    if (hoverLabels.length > 0) {
                        // Check if mouse is over any hover label
                        let mouseOverLabel = false;
                        hoverLabels.forEach(function(label) {
                            const rect = label.getBoundingClientRect();
                            if (rect.width > 0) {
                                mouseOverLabel = true;
                            }
                        });
                        if (!mouseOverLabel) {
                            Plotly.Fx.unhover(plotDiv);
                        }
                    }
                }, 300);
            });
        }, 100);
    })();
    </script>
    """

    fig.write_html(html_path, include_plotlyjs="cdn")

    # Add custom JavaScript to the HTML file
    with open(html_path) as f:
        html_content = f.read()

    # Insert the custom JS before the closing body tag
    html_content = html_content.replace("</body>", custom_js + "</body>")

    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path


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
