"""
Milestone Decomposer – Milestone 2 of Goal 1: Autonomous Goal Executor
Breaks down high-level goals into actionable milestones with effort estimates.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum


class ComplexityLevel(Enum):
    SIMPLE = "simple_linear"
    MODERATE = "phased_iterative"
    COMPLEX = "complex_modular"
    INTENSE = "agile_intensive"


class MilestonePhase(Enum):
    PLANNING = "planning"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REFINEMENT = "refinement"


async def decompose_into_milestones(
    goal_title: str,
    goal_description: str,
    target_date: Optional[str] = None,
    complexity_level: str = "phased_iterative",
    max_milestones: int = 7,
    preferred_session_length_mins: int = 90
) -> Dict[str, Any]:
    """
    Decomposes a goal into actionable milestones with effort estimates.
    
    Args:
        goal_title: Goal name (e.g., "Build Website AI Assistant")
        goal_description: Detailed goal description
        target_date: ISO format deadline (e.g., "2025-02-20")
        complexity_level: "simple_linear", "phased_iterative", "complex_modular", "agile_intensive"
        max_milestones: Max number of milestones to generate (default: 7)
        preferred_session_length_mins: Session duration for effort calculation (default: 90)
    
    Returns:
        {
            "milestones": [...],
            "dependency_graph": {...},
            "risk_assessment": {...},
            "recommendation": "...",
            "success": True/False
        }
    """
    
    try:
        now = datetime.now()
        target = None
        if target_date:
            try:
                target = datetime.fromisoformat(target_date)
            except:
                pass
        
        # Base milestone templates by complexity
        milestone_templates = {
            "simple_linear": [
                ("Planning & Requirements", MilestonePhase.PLANNING, 2),
                ("Implementation", MilestonePhase.IMPLEMENTATION, 5),
                ("Testing & Deployment", MilestonePhase.TESTING, 2),
            ],
            "phased_iterative": [
                ("Planning & Requirements", MilestonePhase.PLANNING, 3),
                ("Design & Architecture", MilestonePhase.DESIGN, 4),
                ("Core Implementation", MilestonePhase.IMPLEMENTATION, 6),
                ("Testing & Refinement", MilestonePhase.TESTING, 3),
                ("Polish & Documentation", MilestonePhase.REFINEMENT, 2),
            ],
            "complex_modular": [
                ("Requirements & Analysis", MilestonePhase.PLANNING, 4),
                ("System Design", MilestonePhase.DESIGN, 5),
                ("Module 1: Core Functionality", MilestonePhase.IMPLEMENTATION, 8),
                ("Module 2: Integration", MilestonePhase.IMPLEMENTATION, 7),
                ("Module 3: Advanced Features", MilestonePhase.IMPLEMENTATION, 6),
                ("Integration Testing", MilestonePhase.TESTING, 5),
                ("Optimization & Release", MilestonePhase.REFINEMENT, 3),
            ],
            "agile_intensive": [
                ("Sprint Planning", MilestonePhase.PLANNING, 2),
                ("Sprint 1: MVP", MilestonePhase.IMPLEMENTATION, 8),
                ("Sprint 2: Features", MilestonePhase.IMPLEMENTATION, 8),
                ("Sprint 3: Refinement", MilestonePhase.IMPLEMENTATION, 6),
                ("Integration & Testing", MilestonePhase.TESTING, 5),
                ("Release Prep", MilestonePhase.REFINEMENT, 3),
            ],
        }
        
        # Get template
        template = milestone_templates.get(complexity_level, milestone_templates["phased_iterative"])
        
        # Consolidate if needed
        milestones = []
        if len(template) > max_milestones:
            # Merge similar phases
            consolidated = {}
            for title, phase, hours in template:
                phase_key = phase.value
                if phase_key not in consolidated:
                    consolidated[phase_key] = (title, phase, hours)
                else:
                    _, p, h = consolidated[phase_key]
                    consolidated[phase_key] = (title, p, h + hours)
            
            milestones = list(consolidated.values())
        else:
            milestones = template
        
        # Build milestone dicts with metadata
        milestone_list = []
        cumulative_hours = 0
        
        for i, (title, phase, hours) in enumerate(milestones, 1):
            sessions_needed = max(1, hours // preferred_session_length_mins)
            
            milestone_list.append({
                "id": f"m{i}",
                "title": title,
                "phase": phase.value,
                "estimated_hours": hours,
                "estimated_sessions": sessions_needed,
                "status": "pending",
                "dependencies": [f"m{j}" for j in range(1, i)] if i > 1 else [],
                "completion_percentage": 0,
                "order": i
            })
            cumulative_hours += hours
        
        # Dependency graph
        dependency_graph = {
            "nodes": [{"id": m["id"], "title": m["title"]} for m in milestone_list],
            "edges": [(m["id"], dep) for m in milestone_list for dep in m["dependencies"]],
            "critical_path": [m["id"] for m in milestone_list],  # Linear dependency
            "total_effort_hours": cumulative_hours
        }
        
        # Risk assessment
        risk_assessment = {
            "timeline_risk": "low",
            "scope_risk": "medium" if len(milestone_list) > 5 else "low",
            "complexity_risk": complexity_level,
            "feasibility_with_deadline": "feasible"
        }
        
        if target:
            days_available = (target - now).days
            sessions_available = (days_available // 7) * (3 + 2)  # ~3 evenings, 2 weekend sessions per week
            
            if cumulative_hours > sessions_available * (preferred_session_length_mins / 60):
                risk_assessment["timeline_risk"] = "high"
                risk_assessment["feasibility_with_deadline"] = "challenging"
        
        # Recommendation
        recommendation = f"Goal decomposed into {len(milestone_list)} milestones. "
        recommendation += f"Estimated total effort: {cumulative_hours} hours (~{sum(m['estimated_sessions'] for m in milestone_list)} sessions). "
        if risk_assessment["timeline_risk"] == "high":
            recommendation += "⚠️ Timeline is tight—prioritize core functionality and defer nice-to-haves."
        else:
            recommendation += "✅ Timeline is achievable with consistent effort."
        
        return {
            "milestones": milestone_list,
            "dependency_graph": dependency_graph,
            "risk_assessment": risk_assessment,
            "recommendation": recommendation,
            "total_estimated_hours": cumulative_hours,
            "success": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "milestones": [],
            "dependency_graph": {},
            "risk_assessment": {},
            "recommendation": "",
            "success": False
        }


def register_milestone_decomposer_tools():
    """Register this tool with the agent's tool system."""
    pass
