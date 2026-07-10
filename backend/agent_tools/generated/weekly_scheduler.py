"""
Weekly Scheduler – Milestone 3 of Goal 1: Autonomous Goal Executor
Allocates milestones to available time slots based on user availability.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TimeSlot:
    """Represents a work session."""
    day: str
    start_time: str
    end_time: str
    duration_mins: int


@dataclass
class AvailabilityProfile:
    """User's recurring availability pattern."""
    weekday_evenings: bool = True  # 6pm-11pm
    weekend_full: bool = True      # 9am-11pm
    sessions_per_week: int = 5     # ~3 weekday + 2 weekend
    session_length_mins: int = 90


async def schedule_milestones(
    milestones: List[Dict[str, Any]],
    start_date: Optional[str] = None,
    weeks_available: int = 4,
    user_availability: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Schedules milestones into user's available time slots.
    
    Args:
        milestones: List of milestone dicts with 'id', 'title', 'estimated_sessions'
        start_date: ISO format start date (default: today)
        weeks_available: Number of weeks to schedule across
        user_availability: Custom availability (default: Gergo's typical)
    
    Returns:
        {
            "schedule": [...],
            "calendar": {...},
            "utilization": {...},
            "recommendations": [...],
            "success": True/False
        }
    """
    
    try:
        # Parse dates
        if not start_date:
            start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = datetime.fromisoformat(start_date)
        
        # Default availability (Gergo's typical schedule)
        if not user_availability:
            availability = AvailabilityProfile()
        else:
            availability = AvailabilityProfile(**user_availability)
        
        # Generate weekly time slots
        slots = []
        current_week = 0
        current_date = start
        
        while current_week < weeks_available:
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Weekday evenings (Mon-Fri, 6pm-11pm)
            if day_of_week < 5 and availability.weekday_evenings:
                slots.append(TimeSlot(
                    day=current_date.strftime("%A %Y-%m-%d"),
                    start_time="18:00",
                    end_time="23:00",
                    duration_mins=300  # 5 hours with breaks
                ))
            
            # Weekend full day (Sat-Sun, 9am-11pm)
            if day_of_week >= 5 and availability.weekend_full:
                slots.append(TimeSlot(
                    day=current_date.strftime("%A %Y-%m-%d"),
                    start_time="09:00",
                    end_time="23:00",
                    duration_mins=840  # 14 hours with breaks
                ))
            
            current_date += timedelta(days=1)
            if current_date.weekday() == 0:  # End of week
                current_week += 1
        
        # Allocate milestones to slots
        schedule = []
        slot_idx = 0
        total_sessions = sum(m.get("estimated_sessions", 1) for m in milestones)
        
        for milestone in milestones:
            sessions_needed = milestone.get("estimated_sessions", 1)
            sessions_scheduled = 0
            
            while sessions_needed > 0 and slot_idx < len(slots):
                slot = slots[slot_idx]
                sessions_in_slot = min(
                    sessions_needed,
                    slot.duration_mins // availability.session_length_mins
                )
                
                if sessions_in_slot > 0:
                    schedule.append({
                        "milestone_id": milestone["id"],
                        "milestone_title": milestone["title"],
                        "scheduled_date": slot.day,
                        "start_time": slot.start_time,
                        "duration_mins": sessions_in_slot * availability.session_length_mins,
                        "sessions": sessions_in_slot,
                        "status": "pending"
                    })
                    
                    sessions_needed -= sessions_in_slot
                    sessions_scheduled += sessions_in_slot
                    
                    # Update slot capacity
                    slot.duration_mins -= sessions_in_slot * availability.session_length_mins
                
                if slot.duration_mins < availability.session_length_mins:
                    slot_idx += 1
        
        # Generate week-by-week calendar
        calendar = {}
        current_week = 0
        week_start = start
        
        for i in range(weeks_available):
            week_end = week_start + timedelta(days=6)
            week_key = f"Week {i+1} ({week_start.strftime('%b %d')} - {week_end.strftime('%b %d')})"
            
            week_milestones = [s for s in schedule 
                              if week_start <= datetime.fromisoformat(s["scheduled_date"].split()[1]) <= week_end]
            
            calendar[week_key] = {
                "milestones_scheduled": len(week_milestones),
                "total_hours": sum(s["duration_mins"] for s in week_milestones) / 60,
                "items": week_milestones
            }
            
            week_start = week_end + timedelta(days=1)
        
        # Utilization analysis
        total_available_mins = sum(s.duration_mins for s in slots)
        total_scheduled_mins = sum(s["duration_mins"] for s in schedule)
        utilization_pct = (total_scheduled_mins / total_available_mins * 100) if total_available_mins > 0 else 0
        
        utilization = {
            "total_available_hours": total_available_mins / 60,
            "total_scheduled_hours": total_scheduled_mins / 60,
            "utilization_percentage": round(utilization_pct, 1),
            "status": "healthy" if 30 <= utilization_pct <= 80 else ("overbooked" if utilization_pct > 80 else "underutilized")
        }
        
        # Recommendations
        recommendations = []
        if utilization_pct > 85:
            recommendations.append("⚠️ Schedule is overbooked. Consider deferring some milestones or extending timeline.")
        elif utilization_pct < 25:
            recommendations.append("💡 Schedule has capacity. Consider adding more goals or increasing session intensity.")
        else:
            recommendations.append("✅ Schedule is well-balanced.")
        
        if slot_idx < len(slots):
            recommendations.append(f"💾 {weeks_available - (slot_idx // 5)} weeks of buffer available for overruns.")
        
        return {
            "schedule": schedule,
            "calendar": calendar,
            "utilization": utilization,
            "recommendations": recommendations,
            "total_milestones_scheduled": len(schedule),
            "weeks_covered": weeks_available,
            "success": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "schedule": [],
            "calendar": {},
            "utilization": {},
            "recommendations": [],
            "success": False
        }


def register_weekly_scheduler_tools():
    """Register this tool with the agent's tool system."""
    pass
