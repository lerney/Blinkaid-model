# File: services/detection/emg_detectors/eye_gaze_detector.py
import logging
from typing import Optional

import numpy as np
from datetime import timedelta

from blinkaid_services.common.enums.detection_types import DetectionType
from blinkaid_services.common.models.detection import DetectionModel
from blinkaid_services.common.models.emg import EmgModel
from blinkaid_services.detection.emg_detectors.base_emg_detector import BaseEmgDetector

logger = logging.getLogger(__name__)


class EyeGazeDetector(BaseEmgDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._left_right_gaze_emg_window = []
        self._left_right_gaze_window_timedelta = timedelta(milliseconds=50)
        self._center_gaze_emg_window = []
        self._center_gaze_window_timedelta = timedelta(milliseconds=150)
        self._num_channels = None

    def detect(self, emg_model: EmgModel) -> Optional[DetectionModel]:
        if self._num_channels is None:
            self._num_channels = len(emg_model.data)

        self._roll_emg_windows(emg_model)
        if len(self._left_right_gaze_emg_window) < 5:
            return None

        gaze_left_supporters, gaze_right_supporters, gaze_center_supporters = self.calc_left_right_center_supporters()

        if gaze_center_supporters >= 14:
            return DetectionModel(
                start_time=self._center_gaze_emg_window[0].timestamp,
                end_time=self._center_gaze_emg_window[-1].timestamp + timedelta(milliseconds=100),
                type=DetectionType.GAZE_CENTER,
                confidence=gaze_center_supporters / self._num_channels,
                metadata={}
            )
        if gaze_right_supporters >= 15:
            return DetectionModel(
                start_time=self._left_right_gaze_emg_window[0].timestamp,
                end_time=self._left_right_gaze_emg_window[-1].timestamp + timedelta(milliseconds=100),
                type=DetectionType.GAZE_RIGHT,
                confidence=gaze_right_supporters / self._num_channels,
                metadata={}
            )
        if gaze_left_supporters >= 14:
            return DetectionModel(
                start_time=self._left_right_gaze_emg_window[0].timestamp,
                end_time=self._left_right_gaze_emg_window[-1].timestamp + timedelta(milliseconds=100),
                type=DetectionType.GAZE_LEFT,
                confidence=gaze_left_supporters / self._num_channels,
                metadata={}
            )


    def calc_left_right_center_supporters(self):
        # Calculate the average of the EMG signal for each channel
        left_right_avgs, left_right_slopes = self._window_averages_and_slopes(
            self._left_right_gaze_emg_window)
        center_mins, center_maxs, center_heights, center_first_sample, center_last_sample = \
            self._window_mins_maxs_heights_firsts_lasts(self._center_gaze_emg_window)

        right_votes = self.vote_on_right_gaze(
            averages=left_right_avgs,
            slopes=left_right_slopes)
        left_votes = self.vote_on_left_gaze(
            averages=left_right_avgs,
            slopes=left_right_slopes)
        center_votes = self.vote_on_center_gaze(
            mins=center_mins,
            firsts=center_first_sample,
            lasts=center_last_sample,
            heights=center_heights,
            maxs=center_maxs)

        left_supporters = sum(left_votes)
        right_supporters = sum(right_votes)
        center_supporters = sum(center_votes)
        logger.debug(f"Gaze right supporters: {right_supporters}")
        logger.debug(f"Gaze left supporters: {left_supporters}")
        logger.debug(f"Gaze center supporters: {center_supporters}")

        return left_supporters, right_supporters, center_supporters

    def _window_mins_maxs_heights_firsts_lasts(self, emg_window: list[EmgModel]):
        emg_data = np.array([emg.data for emg in emg_window])
        mins = np.min(emg_data, axis=0)
        maxs = np.max(emg_data, axis=0)
        heights = maxs - mins
        last_sample = emg_window[-1].data
        first_sample = emg_window[0].data
        return mins, maxs, heights, first_sample, last_sample

    def _window_averages_and_slopes(self, emg_window: list[EmgModel]):
        window_emg_data = np.array([emg.data for emg in emg_window])
        channels_slopes = [emg_window[-1].data[i] - emg_window[0].data[i] for i in range(self._num_channels)]
        channels_averages = np.mean(window_emg_data, axis=0)
        return channels_averages, channels_slopes

    def _roll_emg_windows(self, emg_model):
        self._left_right_gaze_emg_window = roll_emg_window(
            self._left_right_gaze_emg_window,
            emg_model,
            self._left_right_gaze_window_timedelta)

        self._center_gaze_emg_window = roll_emg_window(
            self._center_gaze_emg_window,
            emg_model,
            self._center_gaze_window_timedelta)

    def vote_on_right_gaze(self, averages, slopes) -> list[bool]:
        # Let each channel vote for a right gaze
        votes = [False] * self._num_channels

        votes[0] = (-10 < averages[0] < 10) & (-10 < slopes[0] < 10)
        votes[1] = (-10 < averages[1]) & (-10 < slopes[1])
        votes[2] = (10 < averages[2]) & (0 < slopes[2])
        votes[3] = (10 < averages[3]) & (0 < slopes[3])
        votes[4] = (averages[4] < -10) & (slopes[4] < 0)
        votes[5] = (averages[5] < 0) & (slopes[5] < 0)
        votes[6] = (-10 < averages[6]) & (-10 < slopes[6])
        votes[7] = (10 < averages[7]) & (-10 < slopes[7])
        votes[8] = (averages[8] < -10) & (slopes[8] < 0)
        votes[9] = (averages[9] < -10) & (slopes[9] < 0)
        votes[10] = (-10 < averages[10] < 10)  # & (-10 < slopes[10])
        votes[11] = (-10 < averages[11])  # & (-10 < slopes[11])
        votes[12] = (0 < averages[12]) & (0 < slopes[12])
        votes[13] = (averages[13] < -10) & (slopes[13] < 0)
        votes[14] = (averages[14] < -10) & (slopes[14] < 0)
        votes[15] = (-10 < averages[15] < 10)  # & (-10 < slopes[15])

        return votes

    def vote_on_left_gaze(self, averages, slopes) -> list[bool]:
        # Let each channel vote for a left gaze
        votes = [False] * self._num_channels

        votes[0] = (-10 < averages[0] < 10)  # & (0 < slopes[0])
        votes[1] = (10 < averages[1]) & (0 < slopes[1])
        votes[2] = (averages[2] < 0)  # & (0 < slopes[2])
        votes[3] = (averages[3] < 0)  # & (0 < slopes[3])
        votes[4] = (10 < averages[4]) & (0 < slopes[4])
        votes[5] = (10 < averages[5]) & (0 < slopes[5])
        votes[6] = (10 < averages[6]) & (0 < slopes[6])
        votes[7] = (averages[7] < -10) & (slopes[7] < 0)
        votes[8] = (10 < averages[8]) & (0 < slopes[8])
        votes[9] = (10 < averages[9]) & (0 < slopes[9])
        votes[10] = (10 < averages[10])  # & (0 < slopes[10])
        votes[11] = (10 < averages[11]) & (0 < slopes[11])
        votes[12] = (averages[12] < 10)  # & (slopes[12] < 0)
        votes[13] = (10 < averages[13]) & (0 < slopes[13])
        votes[14] = (10 < averages[14]) & (0 < slopes[14])
        votes[15] = (10 < averages[15]) & (0 < slopes[15])

        return votes

    def vote_on_center_gaze(self, mins, firsts, lasts, heights, maxs) -> list[bool]:
        # Let each channel vote for a center gaze
        votes = [False] * self._num_channels

        votes[0] = (-100 < mins[0] < 0) & (mins[0] < lasts[0]) & (mins[0] < firsts[0]) & (maxs[0] < 10)
        votes[1] = (-100 < mins[1] < -10) & (mins[1] < lasts[1]) & (mins[1] < firsts[1]) & (heights[1] > 10) & (
                maxs[1] < 10)
        votes[2] = (-100 < mins[2] < -10) & (mins[2] < lasts[2]) & (mins[2] < firsts[2]) & (heights[2] > 10) & (
                maxs[2] < 10)
        votes[3] = (-100 < mins[3] < -10) & (mins[3] < lasts[3]) & (mins[3] < firsts[3]) & (heights[3] > 10) & (
                maxs[3] < 10)
        votes[4] = True
        votes[5] = (-100 < mins[5] < -10) & (mins[5] < lasts[5]) & (mins[5] < firsts[5]) & (heights[5] > 10) & (
                maxs[5] < 10)
        votes[6] = (-100 < mins[6] < -10) & (mins[6] < lasts[6]) & (mins[6] < firsts[6]) & (heights[6] > 10) & (
                maxs[6] < 10)
        votes[7] = (-100 < mins[7] < -10) & (mins[7] < lasts[7]) & (mins[7] < firsts[7]) & (heights[7] > 10) & (
                maxs[7] < 10)
        votes[8] = True
        votes[9] = (-100 < mins[9] < -10) & (mins[9] < lasts[9]) & (mins[9] < firsts[9]) & (heights[9] > 10) & (
                maxs[9] < 30)
        votes[10] = (-100 < mins[10] < -10) & (mins[10] < lasts[10]) & (mins[10] < firsts[10]) & (heights[10] > 10) & (
                maxs[10] < 30)
        votes[11] = (-100 < mins[11] < -10) & (mins[11] < lasts[11]) & (mins[11] < firsts[11]) & (heights[11] > 10) & (
                maxs[11] < 30)
        votes[12] = (-100 < mins[12] < -10) & (mins[12] < lasts[12]) & (mins[12] < firsts[12]) & (heights[12] > 10) & (
                maxs[12] < 30)
        votes[13] = (-100 < mins[13] < -10) & (mins[13] < lasts[13]) & (mins[13] < firsts[13]) & (heights[13] > 10) & (
                maxs[13] < 30)
        votes[14] = (-100 < mins[14] < -10) & (mins[14] < lasts[14]) & (mins[14] < firsts[14]) & (heights[14] > 10) & (
                maxs[14] < 30)
        votes[15] = (-100 < mins[15] < -10) & (mins[15] < lasts[15]) & (mins[15] < firsts[15]) & (heights[15] > 10) & (
                maxs[15] < 30)

        return votes


def roll_emg_window(emg_window: list[EmgModel], new_sample: EmgModel, window_timedelta: timedelta) -> list[EmgModel]:
    emg_window.append(new_sample)
    return [emg for emg in emg_window if emg.timestamp > new_sample.timestamp - window_timedelta]
