"""
Self-Learning Loop – Milestone 6 of Goal 1: Autonomous Goal Executor
Analyzes execution history to extract learnings and generate improvement rules.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LearningSession:
    """Structured analysis session."""
    analysis_timestamp: str
    sample_size: int
    confidence_level: str  # low (< 30), medium (30-50), high (50+)
    insights: Dict[str, Any]


async def analyze_and_learn(
    execution_logs: List[Dict[str, Any]],
    scheduled_milestones: List[Dict[str, Any]],
    goal_reports: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Analyzes execution history to generate improvement rules.
    
    Args:
        execution_logs: Historical execution records
        scheduled_milestones: Scheduled milestone data
        goal_reports: Weekly progress reports
    
    Returns:
        {
            "effort_insights": {...},
            "pattern_analysis": {...},
            "scheduling_insights": {...},
            "success_factors": [...],
            "improvement_rules": [...],
            "confidence_scores": {...},
            "success": True/False
        }
    """
    
    try:
        if not goal_reports:
            goal_reports = []
        
        sample_size = len(execution_logs)
        
        # Determine confidence level
        if sample_size < 30:
            confidence = "low"
        elif sample_size < 50:
            confidence = "medium"
        else:
            confidence = "high"
        
        # 1. EFFORT ESTIMATION ANALYSIS
        effort_by_phase = {}
        accuracy_by_phase = {}
        
        for log in execution_logs:
            phase = log.get("phase", "general")
            actual_mins = log.get("duration_mins", 0)
            
            if phase not in effort_by_phase:
                effort_by_phase[phase] = []
            effort_by_phase[phase].append(actual_mins)
        
        # Calculate average by phase
        effort_insights = {}
        for phase, times in effort_by_phase.items():
            avg_mins = sum(times) / len(times) if times else 0
            effort_insights[phase] = {
                "average_minutes": round(avg_mins, 1),
                "sessions_count": len(times),
                "variance": round(max(times) - min(times), 1) if len(times) > 1 else 0
            }
        
        # 2. PATTERN ANALYSIS
        blockers_frequency = {}
        common_blocker_pairs = {}
        
        for log in execution_logs:
            blockers = log.get("blockers", [])
            for blocker in blockers:
                blockers_frequency[blocker] = blockers_frequency.get(blocker, 0) + 1
            
            # Track blocker pairs
            if len(blockers) > 1:
                blocker_pair = tuple(sorted(blockers[:2]))
                common_blocker_pairs[blocker_pair] = common_blocker_pairs.get(blocker_pair, 0) + 1
        
        # Sort by frequency
        top_blockers = sorted(blockers_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        pattern_analysis = {
            "top_blockers": [{"issue": b[0], "frequency": b[1]} for b in top_blockers],
            "recurring_pairs": [{"issues": list(pair[0]), "count": pair[1]} 
                               for pair in sorted(common_blocker_pairs.items(), 
                                                key=lambda x: x[1], reverse=True)[:3]],
            "total_sessions_blocked": len([log for log in execution_logs if log.get("blockers")])
        }
        
        # 3. SCHEDULING INSIGHTS
        sessions_by_day = {}
        efficiency_by_day = {}
        
        for log in execution_logs:
            scheduled_date = log.get("scheduled_date", "unknown")
            day_name = scheduled_date.split()[0] if " " in scheduled_date else "unknown"
            
            if day_name not in sessions_by_day:
                sessions_by_day[day_name] = 0
                efficiency_by_day[day_name] = []
            
            sessions_by_day[day_name] += 1
            
            # Efficiency = completion % - (blockers * 10)
            completion = log.get("completion_percentage", 0)
            blocker_count = len(log.get("blockers", []))
            efficiency = completion - (blocker_count * 10)
            efficiency_by_day[day_name].append(efficiency)
        
        scheduling_insights = {
            "sessions_by_day": sessions_by_day,
            "efficiency_by_day": {day: round(sum(effs) / len(effs), 1) 
                                  for day, effs in efficiency_by_day.items() if effs},
            "most_productive_day": max(efficiency_by_day.items(), 
                                       key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0)[0]
                                   if efficiency_by_day else "unknown"
        }
        
        # 4. SUCCESS FACTORS
        success_factors = []
        
        # Analyze completing vs non-completing sessions
        completed_logs = [log for log in execution_logs if log.get("status") == "completed"]
        failed_logs = [log for log in execution_logs if log.get("status") != "completed"]
        
        if completed_logs and failed_logs:
            avg_completion_completed = sum(log.get("completion_percentage", 0) 
                                          for log in completed_logs) / len(completed_logs)
            avg_blockers_failed = sum(len(log.get("blockers", [])) 
                                     for log in failed_logs) / len(failed_logs)
            
            success_factors.append({
                "factor": "Completion rate",
                "description": f"Successfully completed sessions average {avg_completion_completed:.0f}% completion",
                "impact": "high"
            })
            
            success_factors.append({
                "factor": "Blocker reduction",
                "description": f"Failed sessions average {avg_blockers_failed:.1f} blockers vs {pattern_analysis['total_sessions_blocked'] / max(1, len(execution_logs)):.1f} overall",
                "impact": "high"
            })
        
        # 5. IMPROVEMENT RULES
        improvement_rules = []
        
        # Rule 1: Adjust estimates by phase
        for phase, insights in effort_insights.items():
            if insights["sessions_count"] >= 5:
                improvement_rules.append({
                    "rule_id": f"estimate_{phase}",
                    "category": "estimation",
                    "statement": f"For {phase} work, allocate {insights['average_minutes']} minutes instead of standard estimates",
                    "confidence": confidence,
                    "impact": "medium"
                })
        
        # Rule 2: Blocker prevention
        if top_blockers:
            top_issue = top_blockers[0][0]
            improvement_rules.append({
                "rule_id": "prevent_blocker_1",
                "category": "blocker_prevention",
                "statement": f"Preventive: Address '{top_issue}' proactively before session start",
                "confidence": confidence,
                "impact": "high" if top_blockers[0][1] > 5 else "medium"
            })
        
        # Rule 3: Schedule optimization
        if scheduling_insights["most_productive_day"] != "unknown":
            improvement_rules.append({
                "rule_id": "schedule_optimize",
                "category": "scheduling",
                "statement": f"Schedule intensive milestones on {scheduling_insights['most_productive_day']}s for best results",
                "confidence": confidence,
                "impact": "medium"
            })
        
        # Rule 4: Session pacing
        if sample_size >= 10:
            avg_session_length = sum(log.get("duration_mins", 0) for log in execution_logs) / len(execution_logs)
            if avg_session_length > 120:
                improvement_rules.append({
                    "rule_id": "session_length",
                    "category": "pacing",
                    "statement": "Break sessions into 90-min chunks instead of longer blocks for better retention",
                    "confidence": confidence,
                    "impact": "medium"
                })
        
        # Confidence scores
        confidence_scores = {
            "effort_estimation": "high" if sum(effs["sessions_count"] for effs in effort_insights.values()) >= 20 else "low",
            "pattern_analysis": "high" if pattern_analysis["total_sessions_blocked"] >= 10 else "low",
            "scheduling_optimization": "medium" if len(sessions_by_day) >= 5 else "low",
            "overall": confidence
        }
        
        return {
            "effort_insights": effort_insights,
            "pattern_analysis": pattern_analysis,
            "scheduling_insights": scheduling_insights,
            "success_factors": success_factors,
            "improvement_rules": improvement_rules,
            "confidence_scores": confidence_scores,
            "analysis_timestamp": datetime.now().isoformat(),
            "sample_size": sample_size,
            "success": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "effort_insights": {},
            "pattern_analysis": {},
            "scheduling_insights": {},
            "success_factors": [],
            "improvement_rules": [],
            "confidence_scores": {},
            "success": False
        }


def register_self_learning_loop_tools():
    """Register this tool with the agent's tool system."""
    pass
