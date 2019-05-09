"""Code for reading TensorBoard logs."""
import glob
import os
from typing import List, Dict


class LogsReader:
    """A class for reading TensorBoard logs."""
    def __init__(self, logs_directory: str):
        self.directory = logs_directory
        self.logs_dictionary: Dict[str, None] = dict.fromkeys(self.find_log_paths())

    def find_log_paths(self) -> List[str]:
        """Finds all log paths in directory."""
        glob_string = os.path.join(self.directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
        return event_file_names
