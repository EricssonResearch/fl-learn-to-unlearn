"""Modules for reading data."""

from typing import Sequence
import pandas as pd


class SimpleRead:
    """A simple module for reading data based on pandas."""
    name = 'simple read'

    def __init__(self, data_path: str, tasks: Sequence[str]):
        self.data_path = data_path
        self.tasks = tasks

    def read(self, agent_id: int):
        """Call method.
        """
        return pd.read_pickle(self.data_path + agent_id + '.pickle')
