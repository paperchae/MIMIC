import torch
import numpy as np
from Utils.Signal.Peak import Peak
from Utils.Signal.FFT import FFT
from Utils.Signal.AbnormalDetection import Abnormal
from Utils.Signal.Handler import Manipulation as Ma


class HR:
    """
    Calculate heart rate from the PPG signal, using either peak detection or fft
    Input Signal will be diversified with another vital signal (e.g. ECG) in the future
    """

    def __init__(self, input_sig, fs=60.):
        self.input_sig = input_sig
        if type(self.input_sig) != torch.Tensor:
            self.input_sig = torch.from_numpy(self.input_sig.copy())
        if torch.cuda.is_available():
            self.input_sig = self.input_sig.to('cuda')

        self.test_n, self.sig_length = self.input_sig.shape
        self.fs = fs

    def get_hr(self, calc_type: str = 'peak'):  # from raw signal
        """
        Calculate heart rate from the PPG signal using either peak detection or fft
        :param calc_type: 'peak' or 'fft'
        :param fs: sampling rate of signals, very critical
        :return: hr, abnormally high hr indices, abnormally low hr indices
        """
        if calc_type == 'peak':
            # TODO 1: variable hr needs to be fixed (division by zero, through masking)
            # TODO 2: need to be merged with hrv.timehrvanalysis.mean_hr()
            hrv, _ = self.get_hrv()

            length = torch.sum(hrv > 0, dim=-1)
            hr = 60 / (torch.sum(hrv, dim=-1) / length)
            high_idx, low_idx = Abnormal(hr).amplitude_check(high=180, low=30)
            if high_idx is None and low_idx is None:
                return hr
            else:
                print('Abnormal HR detected, please check the indices')
                return hr, high_idx, low_idx

        elif calc_type == 'fft':
            amp, freq = FFT(self.input_sig, fs=self.fs).fft(dc_removal=True)
            hr = freq[torch.argmax(amp, dim=-1)] * 60
            high_idx, low_idx = Abnormal(hr).amplitude_check(high=180, low=30)
            if high_idx is None and low_idx is None:
                return hr
            else:
                print('Abnormal HR detected, please check the indices')
                return hr, high_idx, low_idx

    def get_hrv(self):
        peaks = Peak(self.input_sig).detection(fs=self.fs)
        hrv = torch.diff(peaks) / self.fs
        mask = (hrv < 0).to(torch.long)
        min_val = torch.min(hrv, dim=-1, keepdim=True)[0]
        hrv -= mask * min_val
        hrv = Ma(hrv).trim_mask()

        return hrv, Ma(peaks + (peaks < 0).to(torch.long)).trim_mask()


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv('../MIMIC_Clinical_Database/III/result/pleth.csv').to_numpy()[:,
           1:]
    data = torch.tensor(data[:, :1200])
    print(data.shape)

    hr, high, low = HR(data, fs=60.).get_hr(calc_type='peak')
    print(hr)
    print('Abnormal High: ', hr[high], high)
    print('Abnormal Low: ', hr[low], low)
