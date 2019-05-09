"""Code for reading TensorBoard logs."""
import glob
import os
from typing import List


class Log:
    """A class for reading TensorBoard logs."""
    def __init__(self):
        pass

    @classmethod
    def find_log_paths(cls, logs_directory) -> List[str]:
        """Finds all log paths in directory."""
        glob_string = os.path.join(logs_directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
        return event_file_names
