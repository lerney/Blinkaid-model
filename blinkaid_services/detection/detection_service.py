# File: services/detection/detection_service.py
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from blinkaid_services.common.enums.detector_types import DetectorTypes
from blinkaid_services.common.models.detection import DetectionModel
from blinkaid_services.common.models.emg import EmgModel
from blinkaid_services.common.pubsub import PubSub
from blinkaid_services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from blinkaid_services.detection.emg_detectors.blink_detector_cnn.blink_detector_cnn import BlinkDetectorCNN
from blinkaid_services.detection.emg_detectors.blink_detector_threshold_voting import BlinkDetectorThresholdVoting
from blinkaid_services.detection.emg_detectors.eye_gaze_detector import EyeGazeDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectorEntry:
    detector: BaseEmgDetector
    is_enabled: bool
    last_detection: Optional[DetectionModel] = None
    allow_overlaps: bool = False
    detections_count: int = 0


class DetectionService:
    def __init__(self, pubsub: PubSub, num_channels: int):
        self._pubsub = pubsub
        self._num_channels = num_channels
        self._detectors_table: dict[str: DetectorEntry] = self.initial_detectors_table

    @property
    def initial_detectors_table(self) -> dict[str, DetectorEntry]:
        return {
            DetectorTypes.BLINK_DETECTOR_THRESHOLD_VOTING:
                DetectorEntry(
                    detector=BlinkDetectorThresholdVoting(),
                    is_enabled=True
                ),
            DetectorTypes.EYE_GAZE_DETECTOR:
                DetectorEntry(
                    detector=EyeGazeDetector(),
                    is_enabled=True
                ),
            DetectorTypes.BLINK_DETECTOR_CNN:
                DetectorEntry(
                    detector=BlinkDetectorCNN(),
                    is_enabled=False
                ),
        }

    async def start(self):
        enabled_detector_names = [name for name, entry in self._detectors_table.items() if entry.is_enabled]
        logger.info(f"🔍 Starting EMG detection service with: {enabled_detector_names}")
        self._detection_task = asyncio.create_task(self._pubsub.subscribe(
            channel=PubSub.Channels.EMG,
            message_handler=self._run_detectors,
            message_class=EmgModel))
        logger.debug("🔍 EMG detection service started")

    async def stop(self):
        logger.info("🛑 Stopping EMG detection service")
        self._detection_task.cancel()
        logger.debug("🛑 EMG detection service stopped")

    async def publish_detection(self, detector_name: str, detection: DetectionModel):
        logger.info(f"{detector_name} detected: {detection}")
        await self._pubsub.publish(PubSub.Channels.DETECTIONS, detection)

    def enable_detector(self, detector_name: DetectorTypes):
        if detector_name not in self._detectors_table:
            raise KeyError(f"Detector {detector_name} not found.")
        self._detectors_table[detector_name].is_enabled = True

    def disable_detector(self, detector_name: str):
        if detector_name not in self._detectors_table:
            raise KeyError(f"Detector {detector_name} not found.")
        self._detectors_table[detector_name].is_enabled = False

    def status(self):
        return {detector for name, detector in self._detectors_table.items()}

    async def _run_detectors(self, emg_sample: EmgModel):
        for detector_name, detector_entry in self._detectors_table.items():
            detection = await self._run_detector(detector_entry, emg_sample)
            if detection:
                await self.publish_detection(detector_name, detection)

    async def _run_detector(self, detector_entry: DetectorEntry, emg_sample: EmgModel) -> Optional[DetectionModel]:
        if detector_entry.is_enabled:
            detection = await asyncio.to_thread(detector_entry.detector.detect, emg_sample)
            if detection:
                if detector_entry.allow_overlaps or not (detection.overlaps(detector_entry.last_detection)):
                    detector_entry.detections_count += 1
                    detector_entry.last_detection = detection
                    return detection

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
