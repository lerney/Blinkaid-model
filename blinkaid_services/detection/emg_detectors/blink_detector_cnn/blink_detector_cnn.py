# File: services/detection/emg_detectors/blink_detector_cnn/blink_detector_cnn.py
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from keras.src.saving import load_model

from blinkaid_services.common.enums.detection_types import DetectionType
from blinkaid_services.common.models.detection import DetectionModel
from blinkaid_services.common.models.emg import EmgModel
from blinkaid_services.detection.emg_detectors.base_emg_detector import BaseEmgDetector

logger = logging.getLogger(__name__)

SAMPLING_FREQ: int = 100
DETECTION_WINDOW_SIZE_SEC: float = 0.5
DETECTION_STEP_SIZE_SEC: float = 0.1
DETECTION_WINDOW_SIZE = int(DETECTION_WINDOW_SIZE_SEC * SAMPLING_FREQ)
DETECTION_STEP_SIZE = int(DETECTION_STEP_SIZE_SEC * SAMPLING_FREQ)
ML_MODELS_DIR = Path(__file__).parent


class BlinkDetectorCNN(BaseEmgDetector):
    def __init__(self,
                 channel_to_predict: int = 12,
                 model_path: str = ML_MODELS_DIR / "blink_model_cnn.keras",
                 detection_window_size: int = DETECTION_WINDOW_SIZE,
                 detection_step_size: int = 40,
                 confidence_threshold: float = 0.6,
                 **kwargs):
        super().__init__(**kwargs)
        self.channel_to_predict = channel_to_predict
        self._model = load_model(model_path)  # Load the CNN model
        self._emg_data_buffer: list[EmgModel] = []
        self._emg_data_buffer_size = detection_window_size
        self._detection_step_size = detection_step_size
        self._detection_confidence_threshold = confidence_threshold
        self._steps = 0

    def detect(self, emg_data: EmgModel) -> Optional[DetectionModel]:
        self._steps += 1
        confidence = 0.0

        # Append the new data to the buffer
        self._emg_data_buffer.append(emg_data)
        if len(self._emg_data_buffer) == self._emg_data_buffer_size:

            # Only predict every detection_step_size steps (to speed up the process)
            if self._steps % self._detection_step_size == 0:
                data = np.array([emg.data[self.channel_to_predict] for emg in self._emg_data_buffer])
                data = data.reshape(1, -1)
                confidence = self._model.predict(data)

            self._emg_data_buffer.pop(0)

        if confidence >= self._detection_confidence_threshold:
            detection_time = emg_data.timestamp
            start_time = detection_time - pd.Timedelta(seconds=0.3)
            end_time = detection_time + pd.Timedelta(seconds=0.2)
            type = DetectionType.BLINK
            metadata = {"confidence": float(confidence)}
            return DetectionModel(start_time=start_time,
                                  end_time=end_time,
                                  type=type,
                                  confidence=confidence,
                                  metadata=metadata)
        return None