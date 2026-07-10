"""
Idle Period Detector for Autonomous Background Executor.

Detects when the user is inactive and available for autonomous work.
Monitors: system inactivity, time of day, goal urgency, and last agent interaction.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Windows GetLastInputInfo (if available)
try:
    import ctypes
    GetLastInputInfo = ctypes.windll.user32.GetLastInputInfo
    GetTickCount = ctypes.windll.kernel32.GetTickCount
    WINDOWS_AVAILABLE = True
except (ImportError, AttributeError):
    WINDOWS_AVAILABLE = False


async def detect_idle_period(
    inactivity_threshold_mins: int = 15,
    last_agent_interaction_threshold_mins: int = 20,
    availability_config: Optional[dict] = None
) -> dict:
    """
    Detect if user is currently in an idle/available period for background work.
    
    Args:
        inactivity_threshold_mins: System idle time required (default 15 min)
        last_agent_interaction_threshold_mins: Time since last agent call (default 20 min)
        availability_config: Optional dict with 'safe_hours' and 'safe_days'
                           Default: weekday 6-11 PM, all day weekends
    
    Returns:
        {
            'success': bool,
            'is_idle': bool,
            'system_idle_mins': float,
            'last_interaction_mins': float,
            'is_safe_time': bool,
            'current_time': str (ISO),
            'blockers': [str],
            'recommendation': str,
            'next_available_window': str (ISO) or None
        }
    """
    
    result = {
        'success': False,
        'is_idle': False,
        'system_idle_mins': 0,
        'last_interaction_mins': 0,
        'is_safe_time': False,
        'current_time': datetime.now().isoformat(),
        'blockers': [],
        'recommendation': '',
        'next_available_window': None
    }
    
    try:
        # 1. Check system inactivity (Windows)
        system_idle_mins = 0
        if WINDOWS_AVAILABLE:
            try:
                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
                
                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                
                if GetLastInputInfo(ctypes.byref(lii)):
                    tick_count = GetTickCount()
                    idle_ms = tick_count - lii.dwTime
                    system_idle_mins = idle_ms / (1000 * 60)
                    result['system_idle_mins'] = round(system_idle_mins, 2)
            except Exception as e:
                result['blockers'].append(f"GetLastInputInfo error: {str(e)}")
        else:
            result['blockers'].append("Windows API unavailable; assuming active")
        
        # 2. Check time since last agent interaction
        state_file = Path('outputs/agent_state.json')
        last_interaction_mins = 0
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    last_call_str = state.get('last_agent_call')
                    if last_call_str:
                        last_call = datetime.fromisoformat(last_call_str)
                        last_interaction_mins = (datetime.now() - last_call).total_seconds() / 60
                        result['last_interaction_mins'] = round(last_interaction_mins, 2)
            except Exception as e:
                result['blockers'].append(f"State file read error: {str(e)}")
        else:
            # First run; assume safe
            last_interaction_mins = inactivity_threshold_mins + 1
        
        # 3. Check if time is safe for background work
        now = datetime.now()
        config = availability_config or {
            'safe_hours': (18, 23),
            'safe_days': [4, 5, 6]
        }
        
        # User profile: evenings & weekends
        weekday = now.weekday()
        hour = now.hour
        safe_hours = config.get('safe_hours', (18, 23))
        safe_days = config.get('safe_days', [4, 5, 6])
        
        is_safe_time = False
        if weekday in safe_days:
            is_safe_time = True
        elif safe_hours[0] <= hour < safe_hours[1]:
            is_safe_time = True
        
        result['is_safe_time'] = is_safe_time
        
        # 4. Determine idleness
        is_idle = (
            system_idle_mins >= inactivity_threshold_mins and
            last_interaction_mins >= last_agent_interaction_threshold_mins and
            is_safe_time
        )
        
        result['is_idle'] = is_idle
        result['success'] = True
        
        # 5. Generate blockers and recommendations
        if not is_idle:
            if system_idle_mins < inactivity_threshold_mins:
                result['blockers'].append(
                    f"System active: {round(system_idle_mins, 1)}min < {inactivity_threshold_mins}min"
                )
            if last_interaction_mins < last_agent_interaction_threshold_mins:
                result['blockers'].append(
                    f"Recent interaction: {round(last_interaction_mins, 1)}min < {last_agent_interaction_threshold_mins}min"
                )
            if not is_safe_time:
                result['blockers'].append(
                    f"Outside safe hours: {now.strftime('%A %H:%M')}"
                )
            
            # Estimate next available window
            if not is_safe_time:
                if weekday in safe_days:
                    next_window = now.replace(hour=safe_hours[0], minute=0, second=0, microsecond=0)
                    if next_window <= now:
                        next_window += timedelta(days=1)
                else:
                    next_window = now.replace(hour=safe_hours[0], minute=0, second=0, microsecond=0)
                    if next_window <= now:
                        days_until_safe = (safe_days[0] - weekday) % 7
                        if days_until_safe == 0:
                            days_until_safe = 7
                        next_window = now + timedelta(days=days_until_safe)
                        next_window = next_window.replace(hour=safe_hours[0], minute=0, second=0, microsecond=0)
                result['next_available_window'] = next_window.isoformat()
        
        if is_idle:
            result['recommendation'] = (
                f"IDLE PERIOD DETECTED: Safe to run background work. "
                f"System idle {round(system_idle_mins, 1)}min, last interaction {round(last_interaction_mins, 1)}min ago."
            )
        else:
            result['recommendation'] = (
                f"NOT IDLE: {len(result['blockers'])} blocker(s). "
                f"Next available: {result.get('next_available_window', 'N/A')}"
            )
        
        return result
        
    except Exception as e:
        result['success'] = False
        result['blockers'].append(f"Fatal error: {str(e)}")
        result['recommendation'] = f"Detection failed: {str(e)}"
        return result


def register_idle_period_detector_tools():
    """Register this tool with the agent's tool system."""
    # Registration handled by agent_core
    pass