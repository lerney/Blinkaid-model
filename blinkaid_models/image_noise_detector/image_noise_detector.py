import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample
from blinkaid_services.common.enums.detection_types import DetectionType
from blinkaid_services.common.models.detection import DetectionModel
from blinkaid_services.common.models.emg import EmgModel
from blinkaid_services.detection.emg_detectors.utils import normalize_and_convert_to_image_resize_clip

logger = logging.getLogger(__name__)

# Constants
SAMPLING_FREQ: int = 250
DETECTION_WINDOW_SIZE_SEC: float = 2
DETECTION_STEP_SIZE_SEC: float = 0.5
DETECTION_WINDOW_SIZE = int(DETECTION_WINDOW_SIZE_SEC * SAMPLING_FREQ)
DETECTION_STEP_SIZE = int(DETECTION_STEP_SIZE_SEC * SAMPLING_FREQ)
ML_MODELS_DIR = Path(__file__).parent
image_height = 50
window_size = 500
final_window_size = 100
num_channels = 16
min_val, max_val = -500, 500


class image_noise_detector(BaseEmgDetector):
    def __init__(self,
                 model_path: str = ML_MODELS_DIR / "noise_model_v1_w500to100h50c1.h5",
                 detection_window_size: int = DETECTION_WINDOW_SIZE,
                 detection_step_size: int = 125,
                 confidence_threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self._model = tf.keras.models.load_model(model_path)  # Load the CNN model
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
        if len(self._emg_data_buffer) > self._emg_data_buffer_size:
            self._emg_data_buffer.pop(0)

        # Only predict every detection_step_size steps
        if self._steps % self._detection_step_size == 0 and len(self._emg_data_buffer) == self._emg_data_buffer_size:
            signal_window = np.array([emg.data for emg in self._emg_data_buffer])
            image = normalize_and_convert_to_image_resize_clip(signal_window,image_height,num_channels,final_window_size,min_val,max_val)
            image_input = image.reshape((1, image_height, final_window_size, 1))
            prediction = self._model.predict(image_input)
            confidence = prediction[0][1]

        if confidence >= self._detection_confidence_threshold:
            detection_time = emg_data.timestamp
            start_time = detection_time - pd.Timedelta(seconds=1)
            end_time = detection_time + pd.Timedelta(seconds=1)
            type = DetectionType.NOISE
            metadata = {"confidence": float(confidence)}
            return DetectionModel(start_time=start_time,
                                  end_time=end_time,
                                  type=type,
                                  confidence=confidence,
                                  metadata=metadata)
        return None
