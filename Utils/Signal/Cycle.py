import torch
from Utils.Signal.FFT import FFT


#  TODO: Implementation needed as standalone class for cycle detection

class Cycle:
    def __init__(self, signals, fs=60.):
        self.signals = signals
        self.fs = fs

    def get_cycle_len(self):
        peak_freq = FFT(self.signals).peak_freq()

        cycle_len = torch.round(self.fs / peak_freq)
        bpm = peak_freq * 60
        if 40 < bpm < 140:
            # self.valid_flag = False
            return True, cycle_len, bpm
        else:
            return False, cycle_len, bpm

    def get_cycle(self):
        # TODO 2: need to solve circular import
        from Utils.Signal.Peak import Peak
        peak = Peak(self.signals)
        dbp_idx = peak.detection(valley=True)
        detrended_sig = peak.signals.squeeze()
        cycle_list = []
        cycle_len_list = []
        lr_check_list = []
        dbp_idx = dbp_idx.squeeze()
        dbp_idx = dbp_idx[dbp_idx > 0]

        for i in range(len(dbp_idx) - 1):
            cycle = detrended_sig[dbp_idx[i]:dbp_idx[i + 1]]
            cycle_list.append(cycle)
            cycle_len_list.append(len(cycle))
            lr_check_list.append(abs(cycle[0] - cycle[-1]))
        avg_cycle_len = np.mean(cycle_len_list, dtype=np.int)

        # peak_bpm = (self.fs / avg_cycle_len) * 60
        length_order = np.argsort(np.abs(np.array(cycle_len_list) - avg_cycle_len))
        diff_order = np.argsort(torch.tensor(lr_check_list).cpu().numpy())
        total_order = length_order[np.where((diff_order == length_order) == True)]
        if len(total_order) > 0:
            most_promising_cycle_idx = total_order[0]
        else:
            most_promising_cycle_idx = length_order[0]
        # check if cycle lengths are similar
        # return cycle_list[most_promising_cycle_idx]
        if len(dbp_idx) >= 2:
            return True, cycle_list[most_promising_cycle_idx]
            # if not check_mahalanobis_dis(cycle_len_list):
            #     return False, cycle_list[most_promising_cycle_idx]
            # if 35 < peak_bpm < 140 or 35 < self.fft_bpm < 140:
            #     return True, cycle_list[most_promising_cycle_idx]
            # else:
            #     return False, cycle_list[most_promising_cycle_idx]
        else:
            return False, cycle_list[most_promising_cycle_idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # import torch

    data = pd.read_csv('/home/paperc/PycharmProjects/MIMIC/MIMIC_Clinical_Database/III/result/pleth.csv').to_numpy()[:,
           1:]
    test = torch.tensor(data[0][:1200])

    cycle_len = Cycle(test).get_cycle_len()
    cycle = Cycle(test).get_cycle()
    print('test')
