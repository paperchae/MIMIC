import torch
from Utils.Signal.Handler import Cleansing
from Utils.Signal.AbnormalDetection import check_mahalanobis_dis


def correction(abnormal_indices):
    diff_mean = torch.mean(torch.diff(abnormal_indices).to(torch.float))
    bad_idx = torch.split(torch.where(torch.diff(abnormal_indices) < 0.8 * diff_mean)[0] + 1, 2)
    remove_cnt = 0
    if len(bad_idx[0]) == 0:
        return abnormal_indices
    else:
        for pair in bad_idx:
            if len(pair) == 1:
                if pair != 1:
                    abnormal_indices = torch.cat((abnormal_indices[:pair[0]], abnormal_indices[pair[0] + 1:]))
                return abnormal_indices
            else:
                pair -= remove_cnt
                roi = abnormal_indices[pair[0] - 5:pair[1] + 5]
                roi_pair = [5, 6]
                std_1 = torch.std(
                    torch.diff(torch.cat((roi[:roi_pair[0]], roi[roi_pair[0] + 1:]))).to(torch.float))
                std_2 = torch.std(
                    torch.diff(torch.cat((roi[:roi_pair[1]], roi[roi_pair[1] + 1:]))).to(torch.float))
                if std_1 < std_2:
                    abnormal_indices = torch.cat((abnormal_indices[:pair[0]], abnormal_indices[pair[0] + 1:]))
                else:
                    abnormal_indices = torch.cat((abnormal_indices[:pair[1]], abnormal_indices[pair[1] + 1:]))
                remove_cnt += 1

        return abnormal_indices


class Peak:
    """
    Peak Detection class
    Supports Peak Detection & Peak Correction
    """

    def __init__(self, signals):
        if type(signals) != torch.Tensor:
            self.signals = torch.tensor(signals)
        if torch.cuda.is_available():
            self.signals = torch.tensor(signals).to('cuda')

    def detection(self, fs=30., detrend=True, valley=False):
        """
        Peak Detection algorithm using pytorch max_pool1d & detrend
        :param fs: sampling rate of signals
        :param detrend: if True, detrend the signals
        :return: Peak indices of signals
        """
        if valley:
            self.signals = -self.signals
        if detrend:
            self.signals = Cleansing(self.signals).detrend()
        if self.signals.dim() == 1:
            self.signals = self.signals.unsqueeze(0)
        test_n, sig_length = self.signals.shape

        index_list = -torch.ones((test_n, int(sig_length // fs) * 5)).to('cuda')
        width = 11  # odd
        window_maxima = torch.nn.functional.max_pool1d(self.signals, width,
                                                       1, padding=width // 2, return_indices=True)[1].to('cuda')
        window_minima = torch.nn.functional.max_pool1d(-self.signals, width,
                                                       1, padding=width // 2, return_indices=True)[1].to('cuda')

        for i in range(test_n):
            peak_candidate = window_maxima[i].unique()
            nice_peak = peak_candidate[window_maxima[i][peak_candidate] == peak_candidate]
            valley_candidate = window_minima[i].unique()
            nice_valley = valley_candidate[window_minima[i][valley_candidate] == valley_candidate]
            # thresholding
            if valley:
                nice_peak = nice_peak[
                    self.signals[i][nice_peak] < torch.mean(self.signals[i][nice_peak]) * 0.8]  # peak thresholding
                nice_valley = nice_valley[
                    self.signals[i][nice_valley] > torch.mean(self.signals[i][nice_valley]) * 1.2]  # min thresholding
            else:
                nice_peak = nice_peak[
                    self.signals[i][nice_peak] > torch.mean(self.signals[i][nice_peak]) * 0.8]  # peak thresholding
                nice_valley = nice_valley[
                    self.signals[i][nice_valley] < torch.mean(self.signals[i][nice_valley]) * 1.2]  # min thresholding
            if len(nice_peak) / len(nice_valley) > 1.8:  # remove false peaks
                if torch.all(nice_peak[:2] > nice_valley[0]):
                    nice_peak = nice_peak[0::2]
                else:
                    nice_peak = nice_peak[1::2]
            maha_flag, pnt = check_mahalanobis_dis(torch.diff(nice_peak), threshold=2)
            if maha_flag:
                nice_peak = correction(nice_peak)
            # beat_interval = torch.diff(nice_peak)  # sample
            # hrv = beat_interval / fs  # second
            # hr = torch.mean(60 / hrv)
            # hr_list[i] = hr
            # hrv_list[i, :len(hrv)] = hrv * 1000  # milli second
            index_list[i, :len(nice_peak)] = nice_peak
        if valley:
            self.signals = -self.signals
        return index_list.to(torch.long)

    def watch_peaks(self, index_list, idx, data_range):
        start_idx, end_idx = data_range
        raw_signals = self.signals[idx].cpu().numpy()
        index_list = index_list.to(torch.int).cpu().numpy()
        index_list = index_list[np.logical_and(index_list > start_idx, index_list < end_idx)]
        # plt.title(str(len(index_list)))
        plt.plot(np.arange(start_idx, end_idx, 1), raw_signals[start_idx:end_idx], color='royalblue')
        plt.plot(index_list, raw_signals[index_list], 'rx')
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    data = pd.read_csv('../MIMIC_Clinical_Database/III/result/pleth.csv').to_numpy()[:, 1:]
    peak_idx = Peak(data).detection(fs=60.)

    detrended = Cleansing(data).detrend()
    detrended_peak_idx = Peak(detrended).detection(fs=60.)
    for i in range(len(peak_idx)):
        if len(peak_idx[i][peak_idx[i] > 0]) != len(detrended_peak_idx[i][detrended_peak_idx[i] > 0]):
            for j in range(8):
                temp_peak = peak_idx[i].cpu().numpy()
                peak_n = len(temp_peak[np.logical_and(temp_peak > j * 2000, temp_peak < (j + 1) * 2000)])
                temp_detrend_peak = detrended_peak_idx[i].cpu().numpy()
                detrend_peak_n = len(
                    temp_detrend_peak[np.logical_and(temp_detrend_peak > j * 2000, temp_detrend_peak < (j + 1) * 2000)])
                if peak_n < detrend_peak_n:
                    plt.title('Raw PPG Signal Peaks: ' + str(peak_n))
                    Peak(data).watch_peaks(peak_idx[i], i, [j * 2000, (j + 1) * 2000])
                    plt.title('Detrended PPG Signal Peaks: ' + str(detrend_peak_n))
                    Peak(detrended).watch_peaks(detrended_peak_idx[i], i, [j * 2000, (j + 1) * 2000])
                    print('break')
        else:
            continue
