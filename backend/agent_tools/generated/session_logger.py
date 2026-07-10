"""
Session Logger for Autonomous Background Executor.

Records autonomous work sessions, captures outcomes, errors, findings,
and generates recommendations for future sessions.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


async def log_session(
    session_id: str,
    task_name: str,
    milestone_id: str,
    goal_id: str,
    status: str,
    duration_mins: float,
    result_data: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    findings: Optional[List[str]] = None,
    blockers_discovered: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None
) -> dict:
    """
    Log a completed autonomous work session.
    
    Records what was attempted, outcomes, errors, new blockers, and recommendations
    for improving future sessions.
    
    Args:
        session_id: Unique session identifier (e.g. 'bg_20250108_001')
        task_name: What was worked on
        milestone_id: Associated milestone UUID
        goal_id: Associated goal UUID
        status: Task status (COMPLETED, TIMEOUT, FAILED, PARTIAL)
        duration_mins: How long the work took
        result_data: Any structured results from the task
        errors: List of error messages encountered
        findings: List of key findings or discoveries
        blockers_discovered: New blockers uncovered during work
        recommendations: Suggestions for improvement or next steps
    
    Returns:
        {
            'success': bool,
            'session_id': str,
            'timestamp': str (ISO),
            'log_file': str (path where session was saved),
            'summary': str,
            'findings_count': int,
            'blockers_discovered_count': int,
            'recommendations_count': int,
            'health_score': float (0-100)
        }
    """
    
    result = {
        'success': False,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'log_file': None,
        'summary': '',
        'findings_count': 0,
        'blockers_discovered_count': 0,
        'recommendations_count': 0,
        'health_score': 0.0
    }
    
    try:
        # 1. Prepare session record
        session_record = {
            'session_id': session_id,
            'timestamp': result['timestamp'],
            'goal_id': goal_id,
            'milestone_id': milestone_id,
            'task_name': task_name,
            'status': status,
            'duration_mins': duration_mins,
            'result_data': result_data or {},
            'errors': errors or [],
            'findings': findings or [],
            'blockers_discovered': blockers_discovered or [],
            'recommendations': recommendations or []
        }
        
        result['findings_count'] = len(findings or [])
        result['blockers_discovered_count'] = len(blockers_discovered or [])
        result['recommendations_count'] = len(recommendations or [])
        
        # 2. Calculate health score
        health_score = 100.0
        if status == 'COMPLETED':
            health_score = 95.0
        elif status == 'PARTIAL':
            health_score = 70.0
        elif status == 'TIMEOUT':
            health_score = 50.0
        elif status == 'FAILED':
            health_score = 20.0
        
        # Adjust by errors
        health_score -= len(errors or []) * 5
        
        # Boost by findings
        health_score += len(findings or []) * 2
        
        # Penalize by blockers
        health_score -= len(blockers_discovered or []) * 3
        
        health_score = max(0, min(100, health_score))
        result['health_score'] = round(health_score, 1)
        
        # 3. Save to session log file
        log_dir = Path('outputs/session_logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'{session_id}.json'
        with open(log_file, 'w') as f:
            json.dump(session_record, f, indent=2)
        
        result['log_file'] = str(log_file)
        
        # 4. Append to master session log
        master_log_file = Path('outputs/session_logs/master.jsonl')
        with open(master_log_file, 'a') as f:
            f.write(json.dumps(session_record) + '\n')
        
        # 5. Update goal/milestone tracking
        goals_file = Path('outputs/goals.json')
        if goals_file.exists():
            try:
                with open(goals_file, 'r') as f:
                    goals = json.load(f)
                
                for goal in goals:
                    if goal.get('id') == goal_id:
                        for milestone in goal.get('milestones', []):
                            if milestone.get('id') == milestone_id:
                                milestone['last_work_session'] = session_id
                                milestone['last_work_time'] = result['timestamp']
                                milestone['session_count'] = milestone.get('session_count', 0) + 1
                                milestone['total_work_mins'] = milestone.get('total_work_mins', 0) + duration_mins
                
                with open(goals_file, 'w') as f:
                    json.dump(goals, f, indent=2)
            except Exception as e:
                pass  # Don't fail if goals.json can't be updated
        
        # 6. Generate summary
        result['summary'] = (
            f"Session {session_id}: {task_name} [{status}] "
            f"Duration: {duration_mins:.1f}min | "
            f"Findings: {result['findings_count']} | "
            f"Blockers: {result['blockers_discovered_count']} | "
            f"Health: {result['health_score']:.1f}/100"
        )
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['success'] = False
        result['summary'] = f'Session logging failed: {str(e)}'
        return result


async def generate_session_report(
    days: int = 7,
    goal_id: Optional[str] = None
) -> dict:
    """
    Generate a summary report of recent background work sessions.
    
    Args:
        days: Look back this many days (default 7 = weekly)
        goal_id: Filter by goal ID. If None, include all.
    
    Returns:
        {
            'success': bool,
            'period': str (e.g. 'Last 7 days'),
            'total_sessions': int,
            'total_work_mins': float,
            'session_count_by_status': dict,
            'average_health_score': float,
            'top_findings': [str],
            'blockers_summary': [str],
            'recommendations_summary': [str],
            'next_focus_areas': [str],
            'report': str
        }
    """
    
    result = {
        'success': False,
        'period': f'Last {days} days',
        'total_sessions': 0,
        'total_work_mins': 0.0,
        'session_count_by_status': {},
        'average_health_score': 0.0,
        'top_findings': [],
        'blockers_summary': [],
        'recommendations_summary': [],
        'next_focus_areas': [],
        'report': ''
    }
    
    try:
        log_dir = Path('outputs/session_logs')
        if not log_dir.exists():
            result['success'] = True
            result['report'] = 'No session logs found yet.'
            return result
        
        # Read master log
        master_log = log_dir / 'master.jsonl'
        if not master_log.exists():
            result['success'] = True
            result['report'] = 'No session logs found yet.'
            return result
        
        sessions = []
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        with open(master_log, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    session = json.loads(line)
                    session_time = datetime.fromisoformat(session['timestamp']).timestamp()
                    if session_time >= cutoff:
                        if goal_id is None or session.get('goal_id') == goal_id:
                            sessions.append(session)
                except:
                    pass
        
        if not sessions:
            result['success'] = True
            result['report'] = f'No sessions in the last {days} days.'
            return result
        
        result['total_sessions'] = len(sessions)
        
        # Aggregate metrics
        total_mins = 0
        status_counts = {}
        health_scores = []
        all_findings = []
        all_blockers = []
        all_recommendations = []
        
        for session in sessions:
            status = session.get('status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
            total_mins += session.get('duration_mins', 0)
            
            # Estimate health (reconstruct)
            health = 100.0
            if status == 'COMPLETED':
                health = 95.0
            elif status == 'PARTIAL':
                health = 70.0
            elif status == 'TIMEOUT':
                health = 50.0
            elif status == 'FAILED':
                health = 20.0
            health -= len(session.get('errors', [])) * 5
            health += len(session.get('findings', [])) * 2
            health -= len(session.get('blockers_discovered', [])) * 3
            health = max(0, min(100, health))
            health_scores.append(health)
            
            all_findings.extend(session.get('findings', []))
            all_blockers.extend(session.get('blockers_discovered', []))
            all_recommendations.extend(session.get('recommendations', []))
        
        result['total_work_mins'] = round(total_mins, 1)
        result['session_count_by_status'] = status_counts
        result['average_health_score'] = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0
        
        # Top findings (most frequent)
        finding_counts = {}
        for finding in all_findings:
            finding_counts[finding] = finding_counts.get(finding, 0) + 1
        result['top_findings'] = sorted(finding_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        result['top_findings'] = [f[0] for f in result['top_findings']]
        
        # Blockers summary
        blocker_counts = {}
        for blocker in all_blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        result['blockers_summary'] = sorted(blocker_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        result['blockers_summary'] = [b[0] for b in result['blockers_summary']]
        
        # Recommendations
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        result['recommendations_summary'] = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        result['recommendations_summary'] = [r[0] for r in result['recommendations_summary']]
        
        # Next focus areas
        if result['blockers_summary']:
            result['next_focus_areas'] = [f"Resolve: {b}" for b in result['blockers_summary'][:3]]
        if result['recommendations_summary']:
            result['next_focus_areas'].extend(result['recommendations_summary'][:2])
        
        # Generate report text
        report = f"""
AUTONOMOUS SESSION REPORT - {result['period']}

Total Sessions: {result['total_sessions']}
Total Work Time: {result['total_work_mins']:.1f} minutes
Average Health Score: {result['average_health_score']:.1f}/100

Status Breakdown: {result['session_count_by_status']}

Top Findings:
{chr(10).join([f"  - {f}" for f in result['top_findings']]) or "  (none)"}

Blockers Identified:
{chr(10).join([f"  - {b}" for b in result['blockers_summary']]) or "  (none)"}

Recommendations:
{chr(10).join([f"  - {r}" for r in result['recommendations_summary']]) or "  (none)"}

Next Focus:
{chr(10).join([f"  - {a}" for a in result['next_focus_areas']]) or "  (none)"}
"""
        result['report'] = report
        result['success'] = True
        return result
        
    except Exception as e:
        result['success'] = False
        result['report'] = f'Report generation failed: {str(e)}'
        return result


def register_session_logger_tools():
    """Register this tool with the agent's tool system."""
    # Registration handled by agent_core
    pass