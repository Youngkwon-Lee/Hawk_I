from typing import Dict, Any

class TimelineService:
    """
    Service for managing patient timeline data and analyzing medication patterns.
    """
    
    def get_patient_timeline(self, patient_id: str) -> Dict[str, Any]:
        """
        Return an explicit unavailable state until verified medication and
        longitudinal motor observations are connected for this patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary containing timeline data and analysis
        """
        return {
            "patient_id": patient_id,
            "available": False,
            "source": "not_connected",
            "timeline": [],
            "pattern": None,
            "recommendations": None,
        }
