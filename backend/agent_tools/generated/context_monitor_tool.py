"""
Context monitoring tool for the agent.
Tracks context usage patterns and logs warnings to memory.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path

async def monitor_context_usage(threshold_percent: int = 70) -> dict:
    """
    Monitor current context usage and log if approaching threshold.
    Returns usage data and logs a warning fact if usage is high.
    
    Args:
        threshold_percent: Warn if context usage exceeds this % (default 70)
    
    Returns:
        dict with context stats and whether warning was logged
    """
    try:
        # Import from agent_core to get context usage
        from agent_core import get_context_estimate
        
        usage = get_context_estimate()
        percent_used = usage.get('percent_used', 0)
        tokens_used = usage.get('tokens_used', 0)
        context_limit = usage.get('context_limit', 200000)
        
        result = {
            'success': True,
            'tokens_used': tokens_used,
            'context_limit': context_limit,
            'percent_used': percent_used,
            'warning_logged': False,
            'message': f'{percent_used}% context used ({tokens_used}/{context_limit} tokens)'
        }
        
        # Log warning if approaching threshold
        if percent_used >= threshold_percent:
            result['warning_logged'] = True
            result['action_needed'] = f'Context at {percent_used}% — consider summarizing or clearing old turns'
            
            # Store warning in memory for future reference
            warning_log = {
                'timestamp': datetime.now().isoformat(),
                'percent_used': percent_used,
                'tokens_used': tokens_used,
                'warning_level': 'high' if percent_used >= 85 else 'medium'
            }
            
            # Append to context_warnings log file
            log_path = Path('memory/context_warnings.json')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(log_path, 'r') as f:
                    warnings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                warnings = []
            
            warnings.append(warning_log)
            with open(log_path, 'w') as f:
                json.dump(warnings[-100:], f, indent=2)  # Keep last 100 warnings
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to monitor context: {e}'
        }


async def get_context_warning_history() -> dict:
    """
    Retrieve the history of context usage warnings.
    Useful for identifying patterns.
    
    Returns:
        dict with warning history and analysis
    """
    try:
        log_path = Path('memory/context_warnings.json')
        
        if not log_path.exists():
            return {
                'success': True,
                'warning_count': 0,
                'history': [],
                'message': 'No warnings logged yet'
            }
        
        with open(log_path, 'r') as f:
            warnings = json.load(f)
        
        if not warnings:
            return {
                'success': True,
                'warning_count': 0,
                'history': [],
                'message': 'No warnings in history'
            }
        
        # Analyze patterns
        avg_percent = sum(w['percent_used'] for w in warnings) / len(warnings)
        max_percent = max(w['percent_used'] for w in warnings)
        high_count = sum(1 for w in warnings if w['warning_level'] == 'high')
        
        return {
            'success': True,
            'warning_count': len(warnings),
            'avg_percent_used': round(avg_percent, 1),
            'max_percent_used': max_percent,
            'high_warnings_count': high_count,
            'recent_warnings': warnings[-5:],  # Last 5
            'pattern': 'High' if high_count > len(warnings) * 0.3 else 'Normal'
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to retrieve warning history: {e}'
        }


def register_context_monitor_tools():
    """Register context monitoring tools."""
    from agent_tools import register_tool
    
    register_tool(
        'monitor_context_usage',
        monitor_context_usage,
        'Monitor current context usage and log warnings if approaching threshold',
        {
            'threshold_percent': {
                'type': 'integer',
                'description': 'Warn if usage exceeds this percentage (default 70)',
                'required': False
            }
        }
    )
    
    register_tool(
        'get_context_warning_history',
        get_context_warning_history,
        'Retrieve history of context usage warnings and identify patterns',
        {}
    )
