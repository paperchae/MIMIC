import torch
import numpy as np

from Utils.Signal.Cycle import Cycle


def check_mahalanobis_dis(x, threshold: float = 2.0):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    x = x.to(torch.float)

    maha_dis = torch.abs(x - torch.mean(x)) / torch.std(x)
    abnormal_pnt = torch.where(maha_dis > threshold)[0]
    if len(abnormal_pnt) > 0:
        return True, abnormal_pnt
    else:
        return False, None


class Abnormal:
    def __init__(self, input_sig):
        self.input_sig = input_sig
        self.cycle = Cycle(input_sig).get_cycle()

    def is_flipped(self):
        """
        Check if the signal is lr-flipped
        :return: Boolean, True if flipped else False
        """
        if torch.argmax(self.cycle) > len(self.cycle) / 2:
            return True
        else:
            return False

    def has_flat(self):
        """
        Check if the signal has flat area (no change in signal)
        :return:
        """
        diff = torch.diff(self.input_sig)
        range1 = torch.where((diff >= 0) & (diff < 1e-10))[0]
        range2 = torch.where((torch.diff(range1) == 1) & torch.diff(range1) > 0)[0]
        flat = range1[range2]

        return len(flat) / len(self.input_sig)

    def is_overdamped(self):
        """
        Check if the Arterial blood pressure signal is over-damped
        :return: Boolean, True if over-damped else False
        """
        pass

    def is_underdamped(self):
        """
        Check if the Arterial blood pressure signal is under-damped
        :return: Boolean, True if under-damped else False
        """
        start_point = torch.argmax(self.cycle)
        end_point = start_point + int(len(self.cycle) * 0.03)

        roi_diff = torch.diff(self.cycle[start_point:end_point])
        if torch.mean(roi_diff) < -5:
            return True
        else:
            return False

    def amplitude_check(self, high: int = 240, low: int = 30):
        high_pnt = torch.where(self.input_sig > high)[0]
        low_pnt = torch.where(self.input_sig < low)[0]

        if len(high_pnt) > 0 or len(low_pnt) > 0:
            return high_pnt, low_pnt
        else:
            return None, None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
