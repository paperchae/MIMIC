import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from Utils.Signal.Handler import Cleansing
from Utils.Signal.FFT import FFT
from InfoExtract.hr import HR
from Visualization.hrv import hrv_report


class TimeHRVAnalysis:
    """
    Time domain HRV Analysis class
    HRV was derived from the PPG signal using Peak Class from Utils.SignalUtils.Peak

    Supports HR, SDNN, RMSSD, NN50, PNN50
    * HR: Heart Rate
    * SDNN: Standard Deviation of Peak Interval
    * RMSSD: Root Mean Square of Successive Differences
    * NN50: Number of Interval Differences of more than 50ms
    * PNN50: Percentage of NN50

    ** Note: input_signals from peak class has padding value of -1 for tensor shape consistency

    """

    def __init__(self, input_signals: torch.tensor, fs: float = 30.):
        # self.input_signals = input_signals  # self.remove_negative(input_signals)
        self.hrv, self.idx = HR(input_signals).get_hrv()
        self.hrv *= 1000  # to milliseconds
        self.fs = fs

        # self.hr = self.mean_hr()
        self.sdnn = self.calculate_sdnn()  # Standard Deviation of Interval
        self.rmssd = self.calculate_rmssd()
        self.nn50 = self.calculate_nn50()
        self.pnn50 = self.calculate_pnn50()
        # self.apen = self.approximate_entropy()
        self.srd = torch.ones_like(self.pnn50) - self.pnn50

    def mean_hr(self):
        """
        Calculate mean heart rate from HRV
        :return: HR
        """
        length = torch.sum(self.hrv > 0, dim=-1)
        masked_sig = self.hrv + (self.hrv < 0).to(torch.int)
        return 60 / ((torch.sum(masked_sig / 1000, dim=-1)) / length)

    def calculate_sdnn(self):
        """
        Calculate Standard Deviation of Peak Interval
        :return:
        """
        cnt = torch.sum(self.hrv > 0, dim=-1)
        mu = torch.sum(self.hrv, dim=-1) / cnt
        sdnn = torch.sqrt(torch.sum((self.hrv - (self.hrv > 0).to(torch.long) * mu.unsqueeze(1)) ** 2, dim=-1) / cnt)

        return sdnn

    def calculate_rmssd(self):
        """
        Calculate Root Mean
        :return:
        """
        mean = torch.sum(torch.square(torch.diff(self.hrv)), dim=-1) / torch.sum(self.hrv > 0, dim=-1)
        return torch.sqrt(mean)

    def calculate_nn50(self):
        """

        :return:
        """
        return torch.sum(torch.diff(self.hrv, dim=-1) >= 50, dim=-1)

    def calculate_pnn50(self):
        """

        :return:
        """
        return self.nn50 / torch.sum(self.hrv > 0, dim=-1)

    def approximate_entropy(self):
        """

        :return:
        """
        apen = []
        # time_series_data = self.hrv[0]
        m = 2
        r_list = [0.2 * torch.std(sig[sig > 0]) for sig in self.hrv]
        # r_list = 0.2 * torch.std(self.hrv, dim=-1)
        N_list = torch.sum(self.hrv > 0, dim=-1)

        for sig, r, N in zip(self.hrv, r_list, N_list):
            sig = sig[sig > 0]
            phi = np.zeros(N - m + 1)
            for i in tqdm(range(N - m + 1)):
                match_counts = 0
                for j in range(N - m + 1):
                    if i == j:
                        continue

                    dist = np.max(np.abs(sig[i:i + m] - sig[j:j + m]).numpy())
                    if dist <= r:
                        match_counts += 1

                phi[i] = match_counts / (N - m + 1)
            apen.append(-np.log(np.mean(phi)))
        return apen

    def report(self):
        plt.title(
            'Time Domain HRV Analysis \nMean HR(bpm): {:.2f} \nSDNN: {:.2f} \nRMSSD: {:.2f} \nNN50: {:.2f} \nPNN50: {:.2f} \nApproximate Entropy: {:.2f}'.format(
                self.hr, self.sdnn, self.rmssd, self.nn50, self.pnn50, self.apen))
        pass


class FrequencyHRVAnalysis:
    """
    Frequency domain HRV Analysis class
    Supports VLF, LF, HF, LF/HF, Normalized LF, Normalized HF
    """

    def __init__(self, input_signals: torch.tensor, fs=125.):
        self.input_signals = input_signals  # if torch.is_tensor(input_signals) else torch.tensor(input_signals)
        _, self.sig_len = self.input_signals.shape
        self.fs = fs
        self.t_power = self.total_power()
        self.vlf_signal, self.vlf_power = self.vlf()
        self.lf_signal, self.lf_power = self.lf()
        self.hf_signal, self.hf_power = self.hf()
        self.rest_signal, self.rest_power = self.rest_f()
        self.norm_lf = self.normalized_lf()
        self.norm_hf = self.normalized_hf()
        self.lf_hf = self.lf_hf_ratio()

    def band_pass_filter(self, signal, f_low, f_high):
        """
        Band pass filter for 1D signal using FFT
        :param signal:
        :param f_low:
        :param f_high:
        :return:
        """
        fft_result, frequencies = FFT(signal).fft()
        bandpass_filter = torch.logical_and(frequencies >= f_low, frequencies <= f_high)
        filtered_fft = fft_result * bandpass_filter
        filtered_signal = torch.fft.irfft(filtered_fft)
        return torch.real(filtered_signal), torch.sum((torch.abs(filtered_fft) / (self.sig_len / 2)) ** 2, dim=-1)

    def total_power(self):
        """

        :return:
        """
        return torch.sum((torch.abs(torch.fft.rfft(self.input_signals)) / (self.sig_len / 2)) ** 2, dim=-1)

    def vlf(self):  # 0.003 - 0.04 Hz
        """

        :return:
        """
        return self.band_pass_filter(self.input_signals, 0.001, 0.04)

    def lf(self):  # 0.04 - 0.15 Hz
        """

        :return:
        """
        return self.band_pass_filter(self.input_signals, 0.04, 0.15)

    def hf(self):  # 0.15 - 0.4 Hz
        """

        :return:
        """
        return self.band_pass_filter(self.input_signals, 0.15, 0.4)

    def rest_f(self):
        """

        :return:
        """
        return self.band_pass_filter(self.input_signals, 0.4, 2000)

    def lf_hf_ratio(self):
        """

        :return:
        """
        return self.lf_power / self.hf_power

    def normalized_lf(self):
        """

        :return:
        """
        return self.lf_power / (self.lf_power + self.hf_power)

    def normalized_hf(self):
        """

        :return:
        """
        return self.hf_power / (self.lf_power + self.hf_power)

    def report(self):
        pass


if __name__ == "__main__":
    FS = 60.
    pleth = pd.read_csv('/home/paperc/PycharmProjects/rppg/cnibp/preprocessing/pleth.csv', header=None).to_numpy()[1:,
            1:]
    info = pd.read_csv('/home/paperc/PycharmProjects/rppg/cnibp/preprocessing/info.csv').to_numpy()[:, 1:]

    freq = FrequencyHRVAnalysis(input_signals=torch.tensor(pleth), fs=FS)
    time = TimeHRVAnalysis(input_signals=torch.tensor(pleth), fs=FS)
    hrv_list, index_list = time.hrv, time.idx
    hr = HR(pleth, FS).get_hr()[0]

    hrv_report(hr, time, freq, index_list, info)
