"""Tests for LogsReader."""

from logs_reader import LogsReader


class TestLogReader:
    """Tests for LogsReader."""
    def test_finds_all_logs_in_directory(self):
        logs_reader = LogsReader('test_resources/example_logs')
        assert len(logs_reader.log_paths) == 12
        assert any('norm_mean abs_plus_one_log_mean_neg' in log_path and 'GAN' in log_path
                   for log_path in logs_reader.log_paths)
        assert not any('DGGAN' in log_path for log_path in logs_reader.log_paths)
