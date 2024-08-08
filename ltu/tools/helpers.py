"""Useful tools."""

from typing import List
from torch.autograd import Variable

EPSILON = 1e-16


def common_elements(list1: list, list2: list) -> List:
    """Find common elements in two lists."""
    return list(set(list1) & set(list2))


def uncommon_elements(list1: list, list2: list) -> List:
    """Find uncommon elements in two lists."""
    if len(list2) > len(list1):
        return list(set(list2) - set(list1))
    else:
        return list(set(list1) - set(list2))


class TorchFunctions(object):
    """ Contains useful functions in torch. Functions are static methods. """
    @staticmethod
    def make_float_variable(x):
        """Making a variable float."""
        return Variable(x).float()

    @staticmethod
    def make_long_variable(x):
        """Making a variable long."""
        return Variable(x).long()
