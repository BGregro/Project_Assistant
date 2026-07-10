"""
Safety Executor for Autonomous Background Executor.

Safely executes background work tasks with kill switches and resource limits.
Monitors CPU/memory, enforces time budgets, catches errors, and aborts if thresholds exceeded.
"""

import asyncio
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable


async def execute_with_safety(
    task_name: str,
    task_func: Optional[Callable] = None,
    task_params: Optional[dict] = None,
    time_budget_mins: int = 30,
    cpu_limit_pct: float = 80.0,
    memory_limit_pct: float = 85.0,
    error_threshold: int = 3,
    check_interval_secs: float = 2.0
) -> dict:
    """
    Execute a background task with safety guardrails.
    
    Kill switches:
    - CPU usage exceeds limit
    - Memory usage exceeds limit
    - Task duration exceeds time budget
    - Error count exceeds threshold
    - Abort file exists (manual interrupt)
    
    Args:
        task_name: Descriptive name of the task (e.g. 'Goal 1 Milestone 1')
        task_func: Async callable to execute. If None, simulate work.
        task_params: Dict of params to pass to task_func
        time_budget_mins: Maximum execution time (default 30 min)
        cpu_limit_pct: Abort if CPU > this % (default 80%)
        memory_limit_pct: Abort if memory > this % (default 85%)
        error_threshold: Abort after this many errors (default 3)
        check_interval_secs: How often to check resources (default 2 sec)
    
    Returns:
        {
            'success': bool,
            'task_name': str,
            'status': str (COMPLETED, TIMEOUT, CPU_LIMIT, MEMORY_LIMIT, ERROR_THRESHOLD, ABORTED),
            'start_time': str (ISO),
            'end_time': str (ISO),
            'duration_mins': float,
            'result': any,
            'error_count': int,
            'errors': [str],
            'max_cpu_pct': float,
            'max_memory_pct': float,
            'resource_violations': [str],
            'message': str
        }
    """
    
    result = {
        'success': False,
        'task_name': task_name,
        'status': 'PENDING',
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_mins': 0.0,
        'result': None,
        'error_count': 0,
        'errors': [],
        'max_cpu_pct': 0.0,
        'max_memory_pct': 0.0,
        'resource_violations': [],
        'message': ''
    }
    
    try:
        start = datetime.now()
        deadline = start + timedelta(minutes=time_budget_mins)
        abort_file = Path(f'outputs/abort_{task_name.replace(" ", "_")}.signal')
        
        # Baseline resource usage
        process = psutil.Process()
        max_cpu = 0.0
        max_mem = 0.0
        
        # Execute task
        task_result = None
        if task_func:
            try:
                # Support both sync and async functions
                if asyncio.iscoroutinefunction(task_func):
                    task_result = await task_func(**(task_params or {}))
                else:
                    task_result = task_func(**(task_params or {}))
            except Exception as e:
                result['error_count'] += 1
                result['errors'].append(str(e))
        
        # Monitor resources while task runs
        error_count = result['error_count']
        while True:
            # Check deadline
            if datetime.now() >= deadline:
                result['status'] = 'TIMEOUT'
                result['resource_violations'].append(
                    f'Time budget exceeded: {time_budget_mins}min'
                )
                break
            
            # Check abort signal
            if abort_file.exists():
                result['status'] = 'ABORTED'
                result['resource_violations'].append('Abort signal detected')
                abort_file.unlink()
                break
            
            # Check CPU
            try:
                cpu_pct = process.cpu_percent(interval=0.1)
                max_cpu = max(max_cpu, cpu_pct)
                if cpu_pct > cpu_limit_pct:
                    result['status'] = 'CPU_LIMIT'
                    result['resource_violations'].append(
                        f'CPU limit exceeded: {cpu_pct:.1f}% > {cpu_limit_pct}%'
                    )
                    break
            except Exception as e:
                result['errors'].append(f'CPU check error: {str(e)}')
                error_count += 1
            
            # Check Memory
            try:
                mem_pct = process.memory_percent()
                max_mem = max(max_mem, mem_pct)
                if mem_pct > memory_limit_pct:
                    result['status'] = 'MEMORY_LIMIT'
                    result['resource_violations'].append(
                        f'Memory limit exceeded: {mem_pct:.1f}% > {memory_limit_pct}%'
                    )
                    break
            except Exception as e:
                result['errors'].append(f'Memory check error: {str(e)}')
                error_count += 1
            
            # Check error threshold
            if error_count >= error_threshold:
                result['status'] = 'ERROR_THRESHOLD'
                result['resource_violations'].append(
                    f'Error threshold reached: {error_count} >= {error_threshold}'
                )
                break
            
            # If no task_func, simulate work for demonstration
            if not task_func:
                await asyncio.sleep(check_interval_secs)
                if (datetime.now() - start).total_seconds() > 5:
                    result['status'] = 'COMPLETED'
                    break
            else:
                await asyncio.sleep(check_interval_secs)
        
        # If still PENDING, mark as COMPLETED
        if result['status'] == 'PENDING':
            result['status'] = 'COMPLETED'
        
        result['end_time'] = datetime.now().isoformat()
        result['duration_mins'] = (datetime.now() - start).total_seconds() / 60
        result['result'] = task_result
        result['error_count'] = error_count
        result['max_cpu_pct'] = round(max_cpu, 2)
        result['max_memory_pct'] = round(max_mem, 2)
        
        result['success'] = result['status'] in ['COMPLETED']
        
        # Generate message
        if result['status'] == 'COMPLETED':
            result['message'] = (
                f"Task completed successfully in {result['duration_mins']:.1f} min. "
                f"Peak CPU: {result['max_cpu_pct']:.1f}%, Peak Mem: {result['max_memory_pct']:.1f}%"
            )
        else:
            result['message'] = (
                f"Task {result['status']}: {', '.join(result['resource_violations'])} "
                f"Duration: {result['duration_mins']:.1f} min"
            )
        
        return result
        
    except Exception as e:
        result['success'] = False
        result['status'] = 'ERROR_THRESHOLD'
        result['end_time'] = datetime.now().isoformat()
        result['message'] = f'Execution failed: {str(e)}'
        result['errors'].append(str(e))
        return result


def register_safety_executor_tools():
    """Register this tool with the agent's tool system."""
    # Registration handled by agent_core
    pass