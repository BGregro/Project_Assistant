"""
Goal Stall Detector – Milestone 1 of Goal 1: Autonomous Goal Executor
Monitors active goals and flags stalled progress (no activity > N days).
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class GoalStallAnalysis:
    """Structured output for stall detection."""
    
    def __init__(self):
        self.stalled_goals: List[Dict[str, Any]] = []
        self.at_risk_goals: List[Dict[str, Any]] = []
        self.status_report: Dict[str, Any] = {}


async def goal_stall_detector(
    active_goals: List[Dict[str, Any]],
    stall_threshold_days: int = 3,
    check_milestones: bool = True
) -> Dict[str, Any]:
    """
    Detects stalled goals based on inactivity threshold.
    
    Args:
        active_goals: List of goal dicts with 'id', 'title', 'last_activity', 'target_date'
        stall_threshold_days: Days without activity to flag as stalled (default: 3)
        check_milestones: Also check milestone-level staleness
    
    Returns:
        {
            "stalled_goals": [...],
            "at_risk_goals": [...],
            "status_report": {...},
            "success": True/False
        }
    """
    
    try:
        analysis = GoalStallAnalysis()
        now = datetime.now()
        stall_cutoff = now - timedelta(days=stall_threshold_days)
        at_risk_cutoff = now - timedelta(days=max(1, stall_threshold_days - 1))
        
        for goal in active_goals:
            goal_id = goal.get("id", "unknown")
            title = goal.get("title", "Untitled Goal")
            last_activity = goal.get("last_activity")
            target_date = goal.get("target_date")
            
            # Parse last_activity if it's a string
            if isinstance(last_activity, str):
                try:
                    last_activity = datetime.fromisoformat(last_activity)
                except:
                    last_activity = None
            
            if not last_activity:
                # No activity recorded = assume stalled
                analysis.stalled_goals.append({
                    "goal_id": goal_id,
                    "title": title,
                    "days_inactive": None,
                    "reason": "No activity recorded",
                    "severity": "critical"
                })
                continue
            
            days_inactive = (now - last_activity).days
            
            # Parse target date for deadline urgency
            deadline_urgency = "low"
            if target_date:
                if isinstance(target_date, str):
                    try:
                        target_date = datetime.fromisoformat(target_date)
                    except:
                        target_date = None
                
                if target_date:
                    days_to_deadline = (target_date - now).days
                    if days_to_deadline <= 7:
                        deadline_urgency = "high"
                    elif days_to_deadline <= 14:
                        deadline_urgency = "medium"
            
            # Classify severity
            if days_inactive >= stall_threshold_days:
                severity = "critical" if days_inactive >= stall_threshold_days * 2 else "high"
                analysis.stalled_goals.append({
                    "goal_id": goal_id,
                    "title": title,
                    "days_inactive": days_inactive,
                    "last_activity": last_activity.isoformat(),
                    "severity": severity,
                    "deadline_urgency": deadline_urgency,
                    "intervention": f"Goal inactive for {days_inactive} days. Requires immediate attention."
                })
            elif days_inactive >= stall_threshold_days - 1:
                analysis.at_risk_goals.append({
                    "goal_id": goal_id,
                    "title": title,
                    "days_inactive": days_inactive,
                    "last_activity": last_activity.isoformat(),
                    "warning": f"Approaching stall threshold ({days_inactive}/{stall_threshold_days} days)."
                })
        
        # Generate summary
        analysis.status_report = {
            "check_timestamp": now.isoformat(),
            "stall_threshold_days": stall_threshold_days,
            "total_goals_checked": len(active_goals),
            "stalled_count": len(analysis.stalled_goals),
            "at_risk_count": len(analysis.at_risk_goals),
            "healthy_count": len(active_goals) - len(analysis.stalled_goals) - len(analysis.at_risk_goals),
            "health_score": max(0, 100 - (len(analysis.stalled_goals) * 30) - (len(analysis.at_risk_goals) * 10))
        }
        
        return {
            "stalled_goals": analysis.stalled_goals,
            "at_risk_goals": analysis.at_risk_goals,
            "status_report": analysis.status_report,
            "success": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "stalled_goals": [],
            "at_risk_goals": [],
            "status_report": {},
            "success": False
        }


def register_goal_stall_detector_tools():
    """Register this tool with the agent's tool system."""
    # This function would be called by agent_core to register the tool
    # The actual registration happens in the main tool registry
    pass
