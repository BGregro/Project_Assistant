"""
Goal Priority Scorer for Autonomous Background Executor.

Scores active goals and milestones to determine which work to prioritize
during autonomous background execution sessions.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List


async def score_goals(
    goals_data: Optional[List[dict]] = None,
    current_time: Optional[str] = None,
    time_budget_mins: int = 120
) -> dict:
    """
    Score all active goals and milestones to prioritize background work.
    
    Scoring factors:
    - Days until deadline (urgent = higher score)
    - Blocker count (fewer blockers = higher score)
    - Milestone complexity (lower = higher priority for quick wins)
    - User urgency flag (manual override multiplier)
    - Success velocity (historical completion rate)
    
    Args:
        goals_data: List of goal dicts. If None, load from outputs/goals.json
        current_time: ISO timestamp. If None, use now.
        time_budget_mins: Available time for background work (default 120 min)
    
    Returns:
        {
            'success': bool,
            'timestamp': str (ISO),
            'total_goals': int,
            'total_milestones': int,
            'ranked_milestones': [
                {
                    'goal_id': str,
                    'goal_title': str,
                    'milestone_title': str,
                    'score': float (0-100),
                    'urgency': str (CRITICAL, HIGH, MEDIUM, LOW),
                    'days_until_deadline': float,
                    'blocker_count': int,
                    'estimated_time_mins': int,
                    'is_achievable_in_budget': bool,
                    'recommendation': str
                }
            ],
            'top_3_milestones': [<same structure>],
            'available_time_mins': int,
            'message': str
        }
    """
    
    result = {
        'success': False,
        'timestamp': current_time or datetime.now().isoformat(),
        'total_goals': 0,
        'total_milestones': 0,
        'ranked_milestones': [],
        'top_3_milestones': [],
        'available_time_mins': time_budget_mins,
        'message': ''
    }
    
    try:
        # 1. Load goals if not provided
        if goals_data is None:
            goals_file = Path('outputs/goals.json')
            if not goals_file.exists():
                result['message'] = 'No goals file found. Create goals first.'
                result['success'] = True
                return result
            
            with open(goals_file, 'r') as f:
                all_goals = json.load(f)
                goals_data = [g for g in all_goals if g.get('status') == 'active']
        
        result['total_goals'] = len(goals_data)
        
        # 2. Parse current time
        now = datetime.fromisoformat(result['timestamp'])
        
        # 3. Score each milestone
        milestone_scores = []
        
        for goal in goals_data:
            goal_id = goal.get('id', 'unknown')
            goal_title = goal.get('title', 'Untitled')
            milestones = goal.get('milestones', [])
            deadline_str = goal.get('target_date')
            
            for milestone in milestones:
                if milestone.get('status') == 'complete':
                    continue  # Skip completed milestones
                
                milestone_title = milestone.get('title', 'Untitled')
                blockers = milestone.get('blockers', [])
                estimated_time = milestone.get('estimated_time_mins', 60)
                urgency_flag = milestone.get('urgency', 'MEDIUM')
                success_rate = milestone.get('success_rate', 0.7)  # Historical
                
                # Calculate days until deadline
                days_until = float('inf')
                if deadline_str:
                    try:
                        deadline = datetime.fromisoformat(deadline_str)
                        days_until = max(0, (deadline - now).days)
                    except:
                        pass
                
                # Scoring logic
                deadline_score = 0
                if days_until == float('inf'):
                    deadline_score = 30  # No deadline
                elif days_until == 0:
                    deadline_score = 100  # Due today
                elif days_until <= 7:
                    deadline_score = 80 + (7 - days_until) * 2.86
                elif days_until <= 30:
                    deadline_score = 50 + (30 - days_until)
                else:
                    deadline_score = 20
                
                blocker_score = max(0, 100 - (len(blockers) * 15))
                
                # Complexity score (prefer quick wins)
                complexity_score = 0
                if estimated_time <= 30:
                    complexity_score = 90
                elif estimated_time <= 60:
                    complexity_score = 70
                elif estimated_time <= 120:
                    complexity_score = 50
                else:
                    complexity_score = 20
                
                # Urgency multiplier
                urgency_mult = {
                    'CRITICAL': 1.5,
                    'HIGH': 1.3,
                    'MEDIUM': 1.0,
                    'LOW': 0.8
                }.get(urgency_flag, 1.0)
                
                # Velocity score
                velocity_score = success_rate * 100
                
                # Composite score (weighted)
                composite = (
                    deadline_score * 0.35 +
                    blocker_score * 0.25 +
                    complexity_score * 0.25 +
                    velocity_score * 0.15
                ) * urgency_mult
                
                composite = min(100, max(0, composite))
                
                # Determine urgency label
                if composite >= 85:
                    urgency_label = 'CRITICAL'
                elif composite >= 70:
                    urgency_label = 'HIGH'
                elif composite >= 50:
                    urgency_label = 'MEDIUM'
                else:
                    urgency_label = 'LOW'
                
                is_achievable = estimated_time <= time_budget_mins
                
                recommendation = ''
                if is_achievable:
                    recommendation = f"Achievable in budget. Est. {estimated_time}min."
                else:
                    recommendation = f"Exceeds {time_budget_mins}min budget. Break into subtasks."
                
                milestone_scores.append({
                    'goal_id': goal_id,
                    'goal_title': goal_title,
                    'milestone_title': milestone_title,
                    'score': round(composite, 1),
                    'urgency': urgency_label,
                    'days_until_deadline': days_until if days_until != float('inf') else None,
                    'blocker_count': len(blockers),
                    'estimated_time_mins': estimated_time,
                    'is_achievable_in_budget': is_achievable,
                    'recommendation': recommendation
                })
        
        result['total_milestones'] = len(milestone_scores)
        
        # 4. Sort by score (descending)
        milestone_scores.sort(key=lambda x: x['score'], reverse=True)
        result['ranked_milestones'] = milestone_scores
        
        # 5. Get top 3
        result['top_3_milestones'] = milestone_scores[:3]
        
        # 6. Generate message
        if not milestone_scores:
            result['message'] = 'No active milestones found.'
        else:
            top = result['top_3_milestones'][0] if result['top_3_milestones'] else None
            if top:
                result['message'] = (
                    f"Prioritized {len(milestone_scores)} milestone(s). "
                    f"Top recommendation: [{top['goal_title']}] {top['milestone_title']} "
                    f"(Score: {top['score']}/100, {top['urgency']})"
                )
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['success'] = False
        result['message'] = f'Scoring failed: {str(e)}'
        return result


def register_goal_priority_scorer_tools():
    """Register this tool with the agent's tool system."""
    # Registration handled by agent_core
    pass