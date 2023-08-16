import torch
import numpy as np


class Abnormal:
    def __init__(self, input_sig):
        self.input_sig = input_sig

    def is_flipped(self):
        pass

    def has_flat(self):
        pass

    def is_overdampped(self):
        pass

    def is_underdampped(self):
        pass
