"""Code for reading TensorBoard logs."""
import glob
import os
from typing import List
import pandas as pd
import tensorflow as tf


class Log:
    """A class for reading TensorBoard logs."""
    def __init__(self, event_file_path: str):
        self.event_file_path: str = event_file_path
        self.scalars_data_frame: pd.DataFrame = self.data_frame_from_event_file_scalar_summaries(self.event_file_path)

    @classmethod
    def find_log_paths(cls, logs_directory) -> List[str]:
        """Finds all log paths in directory."""
        glob_string = os.path.join(logs_directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
        return event_file_names

    @staticmethod
    def data_frame_from_event_file_scalar_summaries(event_file_path: str) -> pd.DataFrame:
        """Creates a Pandas data frame from a scalar summaries event file."""
        summary_iterator = tf.compat.v1.train.summary_iterator(event_file_path)
        scalars_data_frame = pd.DataFrame()
        for event in summary_iterator:
            event_step = int(event.step)
            for value in event.summary.value:
                scalars_data_frame.at[event_step, value.tag] = value.simple_value
        scalars_data_frame.sort_index(inplace=True)
        return scalars_data_frame
