"""Tests for Log."""

from log import Log


class TestLog:
    """Tests for Log."""
    def test_can_find_all_logs_in_directory(self):
        log_paths = Log.find_log_paths('test_resources/example_logs')
        assert len(log_paths) == 12
        assert any('norm_mean abs_plus_one_log_mean_neg' in log_path and 'GAN' in log_path
                   for log_path in log_paths)
        assert not any('DGGAN' in log_path for log_path in log_paths)

