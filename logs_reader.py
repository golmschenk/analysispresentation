"""Code for reading TensorBoard logs."""
import glob
import os
from typing import List


class LogsReader:
    """A class for reading TensorBoard logs."""
    def __init__(self, logs_directory: str):
        self.directory = logs_directory
        self.log_paths: List[str] = self.find_log_paths()

    def find_log_paths(self):
        glob_string = os.path.join(self.directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
        return event_file_names
