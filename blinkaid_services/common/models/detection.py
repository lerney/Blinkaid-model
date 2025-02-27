# File: services/common/models/detection.py
from datetime import datetime

from pydantic import BaseModel

from blinkaid_services.common.enums.detection_types import DetectionType


class DetectionModel(BaseModel):
    start_time: datetime
    end_time: datetime
    type: DetectionType
    confidence: float = 1  # 0-1 value indicating the confidence of the detection (0 = low, 1 = high)
    metadata: dict

    def overlaps(self, other: "DetectionModel") -> bool:
        if other is None:
            return False
        return self.start_time < other.end_time and other.start_time < self.end_time
