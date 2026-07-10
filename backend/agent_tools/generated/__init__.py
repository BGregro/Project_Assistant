"""
Goal 1: Autonomous Goal Executor – 6 integrated tools
Complete implementation of autonomous goal management and execution.

Tools included:
1. goal_stall_detector.py - Detects stalled goals
2. milestone_decomposer.py - Breaks goals into milestones
3. weekly_scheduler.py - Schedules milestones to time slots
4. goal_execution_engine.py - Executes milestones
5. progress_reporter.py - Generates weekly reports
6. self_learning_loop.py - Learns from execution history

Import pattern:
    from backend.agent_tools.generated.goal_stall_detector import goal_stall_detector
    from backend.agent_tools.generated.milestone_decomposer import decompose_into_milestones
    from backend.agent_tools.generated.weekly_scheduler import schedule_milestones
    from backend.agent_tools.generated.goal_execution_engine import execute_next_milestone
    from backend.agent_tools.generated.progress_reporter import generate_weekly_progress_report
    from backend.agent_tools.generated.self_learning_loop import analyze_and_learn
"""

__version__ = "1.0.0"
__author__ = "Autonomous Agent"
__description__ = "Goal 1: Autonomous Goal Executor – Complete Implementation"

# Lazy imports to avoid circular dependencies
def get_all_functions():
    """Returns dict of all tool functions for dynamic registration."""
    return {
        "goal_stall_detector": "backend.agent_tools.generated.goal_stall_detector:goal_stall_detector",
        "decompose_into_milestones": "backend.agent_tools.generated.milestone_decomposer:decompose_into_milestones",
        "schedule_milestones": "backend.agent_tools.generated.weekly_scheduler:schedule_milestones",
        "execute_next_milestone": "backend.agent_tools.generated.goal_execution_engine:execute_next_milestone",
        "execute_goal_session": "backend.agent_tools.generated.goal_execution_engine:execute_goal_session",
        "batch_execute_milestones": "backend.agent_tools.generated.goal_execution_engine:batch_execute_milestones",
        "generate_weekly_progress_report": "backend.agent_tools.generated.progress_reporter:generate_weekly_progress_report",
        "analyze_and_learn": "backend.agent_tools.generated.self_learning_loop:analyze_and_learn",
    }
