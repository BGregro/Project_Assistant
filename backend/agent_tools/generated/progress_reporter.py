"""
Progress Reporter – Milestone 5 of Goal 1: Autonomous Goal Executor
Generates weekly progress summaries with metrics, blockers, and recommendations.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class GoalStatus(Enum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BLOCKED = "blocked"
    STALLED = "stalled"
    COMPLETED = "completed"


class HealthScore(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


async def generate_weekly_progress_report(
    goals: List[Dict[str, Any]],
    execution_logs: Optional[List[Dict[str, Any]]] = None,
    scheduled_milestones: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generates comprehensive weekly progress report for all goals.
    
    Args:
        goals: List of goals with metadata
        execution_logs: Execution history
        scheduled_milestones: Scheduled milestones
    
    Returns:
        {
            "summary": {...},
            "goal_reports": [...],
            "weekly_metrics": {...},
            "blockers": [...],
            "recommendations": [...],
            "next_week_focus": [...],
            "success": True/False
        }
    """
    
    try:
        if not execution_logs:
            execution_logs = []
        if not scheduled_milestones:
            scheduled_milestones = []
        
        now = datetime.now()
        week_start = now - __import__('datetime').timedelta(days=now.weekday())
        
        # Analyze each goal
        goal_reports = []
        total_hours_invested = 0
        total_milestones = 0
        completed_milestones = 0
        
        for goal in goals:
            goal_id = goal.get("id", "unknown")
            title = goal.get("title", "Untitled")
            target_date = goal.get("target_date")
            
            # Get milestones for this goal
            goal_milestones = [m for m in scheduled_milestones 
                              if m.get("goal_id") == goal_id]
            
            # Get execution logs for this goal
            goal_logs = [log for log in execution_logs 
                        if log.get("goal_id") == goal_id or 
                           log.get("milestone_id") in [m.get("id") for m in goal_milestones]]
            
            # Calculate metrics
            total_milestones += len(goal_milestones)
            completed = len([m for m in goal_milestones if m.get("status") == "completed"])
            completed_milestones += completed
            
            completion_pct = (completed / len(goal_milestones) * 100) if goal_milestones else 0
            
            hours_invested = sum(log.get("duration_mins", 0) / 60 for log in goal_logs)
            total_hours_invested += hours_invested
            
            # Determine status
            if completion_pct == 100:
                status = GoalStatus.COMPLETED.value
            elif completion_pct >= 75:
                status = GoalStatus.ON_TRACK.value
            elif completion_pct >= 50:
                status = GoalStatus.AT_RISK.value
            elif any(log.get("blockers") for log in goal_logs):
                status = GoalStatus.BLOCKED.value
            else:
                status = GoalStatus.STALLED.value
            
            # Check deadline
            days_to_deadline = None
            if target_date:
                try:
                    target = datetime.fromisoformat(target_date)
                    days_to_deadline = (target - now).days
                except:
                    pass
            
            goal_reports.append({
                "goal_id": goal_id,
                "title": title,
                "status": status,
                "completion_percentage": round(completion_pct, 1),
                "milestones_completed": completed,
                "total_milestones": len(goal_milestones),
                "hours_invested": round(hours_invested, 1),
                "sessions_count": len(goal_logs),
                "days_to_deadline": days_to_deadline,
                "blockers": list(set([b for log in goal_logs for b in log.get("blockers", [])]))
            })
        
        # Weekly metrics
        completion_rate = (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0
        
        weekly_metrics = {
            "week_start": week_start.isoformat(),
            "week_end": now.isoformat(),
            "total_goals": len(goals),
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "completion_rate": round(completion_rate, 1),
            "total_hours_invested": round(total_hours_invested, 1),
            "execution_sessions": len(execution_logs),
            "average_session_hours": round(total_hours_invested / len(execution_logs), 1) if execution_logs else 0
        }
        
        # Collect all blockers
        all_blockers = []
        for report in goal_reports:
            for blocker in report.get("blockers", []):
                all_blockers.append({
                    "goal_id": report["goal_id"],
                    "goal_title": report["title"],
                    "issue": blocker,
                    "severity": "medium"
                })
        
        # Recommendations
        recommendations = []
        
        blocked_count = len([r for r in goal_reports if r["status"] == "blocked"])
        if blocked_count > 0:
            recommendations.append(f"⚠️ {blocked_count} goal(s) are blocked. Address blockers before proceeding.")
        
        stalled_count = len([r for r in goal_reports if r["status"] == "stalled"])
        if stalled_count > 0:
            recommendations.append(f"📋 {stalled_count} goal(s) are stalled. Check if milestones need decomposition.")
        
        at_risk_count = len([r for r in goal_reports if r["status"] == "at_risk"])
        if at_risk_count > 0:
            recommendations.append(f"⏰ {at_risk_count} goal(s) are at risk. Increase session intensity or extend deadline.")
        
        on_track_count = len([r for r in goal_reports if r["status"] == "on_track"])
        if on_track_count > 0:
            recommendations.append(f"✅ {on_track_count} goal(s) are on track. Maintain current pace.")
        
        if total_hours_invested < 5:
            recommendations.append("💡 Low effort this week. Consider adding more goals or increasing session frequency.")
        
        # Next week focus (priority scoring)
        next_week_focus = []
        for report in goal_reports:
            priority_score = 0
            
            # Higher priority for stalled/blocked
            if report["status"] in ["stalled", "blocked"]:
                priority_score += 50
            elif report["status"] == "at_risk":
                priority_score += 30
            
            # Higher priority for approaching deadline
            if report["days_to_deadline"] is not None:
                if report["days_to_deadline"] <= 7:
                    priority_score += 40
                elif report["days_to_deadline"] <= 14:
                    priority_score += 20
            
            # Lower priority if completed
            if report["status"] == "completed":
                priority_score = 0
            
            if priority_score > 0:
                next_week_focus.append({
                    "goal_id": report["goal_id"],
                    "title": report["title"],
                    "priority_score": priority_score,
                    "reason": f"Status: {report['status']}, {report['completion_percentage']}% complete"
                })
        
        next_week_focus.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Health score
        health_emoji = "🟢"
        health_level = HealthScore.EXCELLENT.value
        
        if completion_rate < 20 or blocked_count >= 2:
            health_emoji = "🔴"
            health_level = HealthScore.CRITICAL.value
        elif completion_rate < 40 or blocked_count >= 1:
            health_emoji = "🟠"
            health_level = HealthScore.POOR.value
        elif completion_rate < 60 or at_risk_count >= 2:
            health_emoji = "🟡"
            health_level = HealthScore.FAIR.value
        elif completion_rate < 80:
            health_emoji = "🟢"
            health_level = HealthScore.GOOD.value
        
        summary = {
            "report_timestamp": now.isoformat(),
            "health_emoji": health_emoji,
            "health_level": health_level,
            "overall_completion": round(completion_rate, 1),
            "goals_summary": {
                "on_track": on_track_count,
                "at_risk": at_risk_count,
                "blocked": blocked_count,
                "stalled": stalled_count,
                "completed": len([r for r in goal_reports if r["status"] == "completed"])
            }
        }
        
        return {
            "summary": summary,
            "goal_reports": goal_reports,
            "weekly_metrics": weekly_metrics,
            "blockers": all_blockers,
            "recommendations": recommendations,
            "next_week_focus": next_week_focus,
            "success": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "summary": {},
            "goal_reports": [],
            "weekly_metrics": {},
            "blockers": [],
            "recommendations": [],
            "next_week_focus": [],
            "success": False
        }


def register_progress_reporter_tools():
    """Register this tool with the agent's tool system."""
    pass
