# File: services/common/models/emg.py
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional



# Pydantic model for a single EMG data sample
class EmgModel(BaseModel):
    timestamp: datetime  # Timestamp of the sample
    data: List[float]  # List of EMG data samples (one per channel)

    def __lt__(self, other):
        return self.timestamp < other.timestamp
