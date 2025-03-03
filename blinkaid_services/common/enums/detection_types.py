# File: services/common/enums/detection_types.py
from enum import Enum


class DetectionType(Enum):
    BLINK = "Blink"
    GAZE_LEFT = "Gaze Left"
    GAZE_RIGHT = "Gaze Right"
    GAZE_CENTER = "Gaze Center"
    FROWN = "Frown"
    NOISE = "Noise"
