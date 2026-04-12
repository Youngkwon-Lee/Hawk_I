from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

class TimelineService:
    """
    Service for managing patient timeline data and analyzing medication patterns.
    """
    
    def get_patient_timeline(self, patient_id: str) -> Dict[str, Any]:
        """
        Get timeline data for a patient.
        Currently returns mock data for demonstration.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary containing timeline data and analysis
        """
        # Generate mock timeline data (24 hours, hourly measurements)
        timeline_data = self._generate_mock_timeline()
        
        # Analyze medication pattern
        pattern = self.analyze_medication_pattern(timeline_data)
        
        # Get recommendations
        recommendations = self.recommend_activity_times(pattern)
        
        return {
            "patient_id": patient_id,
            "timeline": timeline_data,
            "pattern": pattern,
            "recommendations": recommendations
        }
    
    def _generate_mock_timeline(self) -> List[Dict[str, Any]]:
        """Generate mock timeline data for demonstration."""
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timeline = []
        
        # Simulate medication taken at 8 AM and 8 PM
        medication_times = [8, 20]
        
        for hour in range(24):
            time = base_time + timedelta(hours=hour)
            
            # Calculate medication effect (peaks 1-2 hours after medication)
            effect = 0
            for med_time in medication_times:
                hours_since_med = (hour - med_time) % 24
                if hours_since_med <= 4:
                    # Peak effect at 1-2 hours, then gradual decline
                    if hours_since_med <= 2:
                        effect += 0.8 - (hours_since_med * 0.1)
                    else:
                        effect += 0.6 - ((hours_since_med - 2) * 0.15)
            
            # Add some randomness
            effect = min(1.0, max(0.0, effect + random.uniform(-0.1, 0.1)))
            
            # Calculate motor score (inverse of effect, 0-100 scale)
            motor_score = int((1 - effect) * 60 + 20)  # Range: 20-80
            
            # Determine ON/OFF state
            state = "ON" if effect > 0.5 else "OFF"
            
            timeline.append({
                "time": time.isoformat(),
                "hour": hour,
                "motor_score": motor_score,
                "medication_effect": round(effect, 2),
                "state": state,
                "tremor_intensity": round((1 - effect) * 5, 1),  # 0-5 scale
                "rigidity": round((1 - effect) * 4, 1),  # 0-4 scale
                "bradykinesia": round((1 - effect) * 3, 1)  # 0-3 scale
            })
        
        return timeline
    
    def analyze_medication_pattern(self, timeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze medication effect pattern from timeline data.
        
        Args:
            timeline_data: List of hourly measurements
            
        Returns:
            Pattern analysis results
        """
        on_periods = []
        off_periods = []
        current_period = None
        
        for i, point in enumerate(timeline_data):
            if point["state"] == "ON":
                if current_period is None or current_period["state"] == "OFF":
                    current_period = {
                        "state": "ON",
                        "start_hour": point["hour"],
                        "end_hour": point["hour"]
                    }
                else:
                    current_period["end_hour"] = point["hour"]
            else:
                if current_period and current_period["state"] == "ON":
                    on_periods.append(current_period)
                    current_period = {
                        "state": "OFF",
                        "start_hour": point["hour"],
                        "end_hour": point["hour"]
                    }
                elif current_period:
                    current_period["end_hour"] = point["hour"]
        
        # Add last period
        if current_period:
            if current_period["state"] == "ON":
                on_periods.append(current_period)
            else:
                off_periods.append(current_period)
        
        # Calculate average scores
        avg_motor_score = sum(p["motor_score"] for p in timeline_data) / len(timeline_data)
        on_avg = sum(p["motor_score"] for p in timeline_data if p["state"] == "ON") / max(1, sum(1 for p in timeline_data if p["state"] == "ON"))
        off_avg = sum(p["motor_score"] for p in timeline_data if p["state"] == "OFF") / max(1, sum(1 for p in timeline_data if p["state"] == "OFF"))
        
        return {
            "on_periods": on_periods,
            "off_periods": off_periods,
            "avg_motor_score": round(avg_motor_score, 1),
            "on_avg_score": round(on_avg, 1),
            "off_avg_score": round(off_avg, 1),
            "total_on_hours": sum((p["end_hour"] - p["start_hour"] + 1) for p in on_periods),
            "total_off_hours": sum((p["end_hour"] - p["start_hour"] + 1) for p in off_periods)
        }
    
    def recommend_activity_times(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate activity recommendations based on medication pattern.
        
        Args:
            pattern: Pattern analysis results
            
        Returns:
            Activity recommendations
        """
        on_periods = pattern["on_periods"]
        
        if not on_periods:
            return {
                "optimal_exercise_times": [],
                "next_medication_time": "8:00 AM",
                "avoid_activities": ["All day - consult doctor"]
            }
        
        # Find longest ON period for exercise
        longest_on = max(on_periods, key=lambda p: p["end_hour"] - p["start_hour"])
        optimal_hour = (longest_on["start_hour"] + longest_on["end_hour"]) // 2
        
        # Format recommendations
        optimal_times = []
        for period in on_periods:
            if period["end_hour"] - period["start_hour"] >= 2:  # At least 2 hours
                start = f"{period['start_hour']:02d}:00"
                end = f"{period['end_hour']:02d}:00"
                optimal_times.append(f"{start} - {end}")
        
        return {
            "optimal_exercise_times": optimal_times,
            "best_exercise_hour": f"{optimal_hour:02d}:00",
            "next_medication_time": "8:00 AM / 8:00 PM",
            "avoid_activities": [
                f"{p['start_hour']:02d}:00 - {p['end_hour']:02d}:00" 
                for p in pattern["off_periods"] 
                if p["end_hour"] - p["start_hour"] >= 2
            ]
        }
