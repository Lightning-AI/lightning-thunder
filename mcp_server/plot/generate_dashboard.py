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
    custom_js = """
    <style>
    /* Make hover labels stay visible and interactive */
    .hoverlayer {
        pointer-events: auto !important;
    }
    .hoverlayer .hovertext {
        pointer-events: auto !important;
    }
    .hoverlayer .hovertext path {
        pointer-events: auto !important;
    }
    .hoverlayer .hovertext text {
        pointer-events: auto !important;
    }
    /* Style links in tooltips */
    .hovertext a {
        color: #3B82F6 !important;
        text-decoration: underline !important;
        cursor: pointer !important;
        pointer-events: auto !important;
    }
    .hovertext a:hover {
        color: #1D4ED8 !important;
        font-weight: bold;
    }
    /* Increase hover area slightly */
    .hoverlayer .hovertext {
        padding: 8px !important;
    }
    </style>
    <script>
    // Make tooltips sticky and clickable - Enhanced version
    (function() {
        let isMouseOverTooltip = false;
        let isMouseOverPoint = false;
        let currentHoverData = null;
        let hideTimeout = null;

        // Wait for plotly to fully load
        setTimeout(function() {
            const plotDiv = document.querySelector('.plotly');
            if (!plotDiv) {
                console.error('Plotly div not found');
                return;
            }

            console.log('Initializing sticky tooltips...');

            // Function to check if mouse is over tooltip area
            function checkTooltipHover() {
                const hoverLayer = document.querySelector('.hoverlayer');
                if (!hoverLayer) return false;

                const hoverTexts = hoverLayer.querySelectorAll('.hovertext');
                for (let hoverText of hoverTexts) {
                    const rect = hoverText.getBoundingClientRect();
                    // Check if tooltip exists and is visible
                    if (rect.width > 0 && rect.height > 0) {
                        return true;
                    }
                }
                return false;
            }

            // Function to make links in tooltips clickable
            function makeLinksClickable() {
                const links = document.querySelectorAll('.hovertext a');
                links.forEach(function(link) {
                    link.style.pointerEvents = 'auto';
                    link.style.cursor = 'pointer';

                    // Remove any existing click handlers to avoid duplicates
                    const newLink = link.cloneNode(true);
                    link.parentNode.replaceChild(newLink, link);

                    // Add click handler
                    newLink.addEventListener('click', function(event) {
                        event.preventDefault();
                        event.stopPropagation();
                        const href = newLink.getAttribute('href');
                        if (href) {
                            window.open(href, '_blank');
                        }
                    });
                });
            }

            // Function to force tooltip to stay visible
            function keepTooltipVisible() {
                if (currentHoverData && (isMouseOverTooltip || isMouseOverPoint)) {
                    // Keep tooltip visible
                    return true;
                }
                return false;
            }

            // Handle hover events
            plotDiv.on('plotly_hover', function(data) {
                console.log('Hover detected on point');
                isMouseOverPoint = true;
                currentHoverData = data;

                clearTimeout(hideTimeout);

                // Make links clickable after a short delay to ensure tooltip is rendered
                setTimeout(function() {
                    makeLinksClickable();

                    // Add mouse event listeners to tooltip
                    const hoverLayer = document.querySelector('.hoverlayer');
                    if (hoverLayer) {
                        hoverLayer.addEventListener('mouseenter', function() {
                            console.log('Mouse entered tooltip area');
                            isMouseOverTooltip = true;
                            clearTimeout(hideTimeout);
                        });

                        hoverLayer.addEventListener('mouseleave', function() {
                            console.log('Mouse left tooltip area');
                            isMouseOverTooltip = false;
                            isMouseOverPoint = false;

                            // Delay hiding to give user time to move back
                            hideTimeout = setTimeout(function() {
                                if (!isMouseOverTooltip && !isMouseOverPoint) {
                                    console.log('Hiding tooltip');
                                    Plotly.Fx.unhover(plotDiv);
                                    currentHoverData = null;
                                }
                            }, 300);
                        });
                    }
                }, 100);
            });

            // Handle unhover events
            plotDiv.on('plotly_unhover', function(data) {
                console.log('Unhover event triggered');

                // Don't immediately hide if mouse is over tooltip
                setTimeout(function() {
                    if (isMouseOverTooltip) {
                        console.log('Keeping tooltip visible - mouse over tooltip');
                        // Re-trigger hover to keep tooltip visible
                        if (currentHoverData) {
                            Plotly.Fx.hover(plotDiv, currentHoverData.points.map(function(pt) {
                                return {
                                    curveNumber: pt.curveNumber,
                                    pointNumber: pt.pointNumber
                                };
                            }));
                        }
                    } else {
                        isMouseOverPoint = false;
                        hideTimeout = setTimeout(function() {
                            if (!isMouseOverTooltip && !isMouseOverPoint) {
                                console.log('Hiding tooltip after delay');
                                currentHoverData = null;
                            }
                        }, 300);
                    }
                }, 50);
            });

            // Handle click events to open GitHub directly
            plotDiv.on('plotly_click', function(data) {
                console.log('Click detected on point');
                if (data.points && data.points.length > 0) {
                    const point = data.points[0];
                    if (point.customdata) {
                        window.open(point.customdata, '_blank');
                    }
                }
            });

            // Global mouse move handler to track position
            document.addEventListener('mousemove', function(e) {
                const hoverLayer = document.querySelector('.hoverlayer');
                if (hoverLayer) {
                    const rect = hoverLayer.getBoundingClientRect();
                    const isOverHoverLayer = (
                        e.clientX >= rect.left &&
                        e.clientX <= rect.right &&
                        e.clientY >= rect.top &&
                        e.clientY <= rect.bottom
                    );

                    if (isOverHoverLayer && checkTooltipHover()) {
                        if (!isMouseOverTooltip) {
                            console.log('Mouse detected over tooltip via mousemove');
                            isMouseOverTooltip = true;
                            clearTimeout(hideTimeout);
                        }
                    }
                }
            });

            console.log('Sticky tooltips initialized successfully');
        }, 500);  // Increased delay to ensure Plotly is fully loaded
    })();
    </script>
    """

    # Write HTML with config to improve hover behavior
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": "pr_dashboard", "height": 800, "width": 1200, "scale": 2},
    }

    fig.write_html(html_path, include_plotlyjs="cdn", config=config)

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
