# File: services/common/enums/detector_types.py
from enum import StrEnum, auto


class DetectorTypes(StrEnum):
    BLINK_DETECTOR_THRESHOLD_VOTING = auto()
    BLINK_DETECTOR_CNN = auto()
    EYE_GAZE_DETECTOR = auto()