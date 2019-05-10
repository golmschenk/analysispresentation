"""Tests for Log."""
import pytest

from log import Log
import math


class TestLog:
    """Tests for Log."""
    def test_can_find_all_logs_in_directory(self):
        log_paths = Log.find_log_paths('test_resources/example_logs')
        assert len(log_paths) == 14
        assert any('norm_mean abs_plus_one_log_mean_neg' in log_path and 'GAN' in log_path
                   for log_path in log_paths)
        assert not any('DGGAN' in log_path for log_path in log_paths)

    def test_can_find_only_event_file_in_directory(self):
        log = Log('test_resources/example_logs/example_log/DNN')
        assert log is not None
        with pytest.raises(AssertionError):  # Only allow a single event file to be contained.
            Log('test_resources/example_logs/example_log')

    def test_can_retrieve_scalar_data_frame(self):
        log = Log('test_resources/example_logs/example_log/DNN/events.out.tfevents.test')
        assert '2_Train_Error/MAE' in log.scalars_data_frame.columns
        assert math.isclose(log.scalars_data_frame['2_Train_Error/MAE'].iloc[0], 0.0297654513)
