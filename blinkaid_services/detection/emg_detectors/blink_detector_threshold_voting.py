# File: services/detection/emg_detectors/blink_detector_threshold_voting.py
import logging
from typing import Optional

import numpy as np
from datetime import timedelta

from blinkaid_services.common.enums.detection_types import DetectionType
from blinkaid_services.common.models.detection import DetectionModel
from blinkaid_services.common.models.emg import EmgModel
from blinkaid_services.detection.emg_detectors.base_emg_detector import BaseEmgDetector

logger = logging.getLogger(__name__)


class BlinkDetectorThresholdVoting(BaseEmgDetector):

    def __init__(self,
                 majority: int = 5,
                 threshold_uV: float = 100,
                 duration_ms=400,
                 detection_delay_ms=100,
                 **kwargs):
        super().__init__(**kwargs)
        self._majority = majority
        self._threshold_uV = threshold_uV
        self._duration_ms = duration_ms
        self._detection_delay_ms = detection_delay_ms
        self._num_channels = None
        self._running_average = None
        self._processed_counter = 0

    @property
    def channel_averages(self):
        return {i: self._running_average[i] for i in range(self._num_channels)}

    def detect(self, emg_model: EmgModel) -> Optional[DetectionModel]:
        if self._num_channels is None:
            self._num_channels = len(emg_model.data)
            self._running_average = np.zeros(self._num_channels)
            self._majority = min(self._majority, self._num_channels)

        voters = 0
        n = self._processed_counter
        self._processed_counter += 1
        now = emg_model.timestamp

        for i, sample in enumerate(emg_model.data):
            if abs(sample - self._running_average[i]) > abs(self._threshold_uV):
                logger.debug(f"Channel {i} voted 'blink'!")
                voters += 1
            self._running_average[i] = (n * self._running_average[i] + sample) / (n + 1)

        if voters >= self._majority:
            logger.debug(f"A {voters}-majority voted 'blink'!")
            confidence = voters / self._num_channels
            duration = timedelta(milliseconds=self._duration_ms)
            start = now - timedelta(milliseconds=self._detection_delay_ms)
            end = now + duration
            return DetectionModel(
                start_time=start,
                end_time=end,
                type=DetectionType.BLINK,
                confidence=confidence,
                metadata={
                    "strength": sum(emg_model.data) / len(emg_model.data),
                    "voters": voters
                })

        return None
