import numpy as np
class Cycle:
    def __init__(self, signals):
        self.signals = signals

    def get_cycle_len(self):
        s_fft = np.fft.fft(self.signals)
        amplitude = (abs(s_fft) * (2 / len(s_fft)))[1:]
        frequency = np.fft.fftfreq(len(s_fft), 1 / self.fs)

        # fft_freq = frequency.copy()
        peak_index = amplitude[:int(len(amplitude) / 2)].argsort()[::-1][:2]
        peak_freq = frequency[peak_index[0]]
        if peak_freq <= 0:
            peak_freq = frequency[peak_index[1]]

        # fft_1x = s_fft.copy()
        # fft_1x[fft_freq != peak_freq] = 0
        # filtered_data = 2*np.fft.ifft(fft_1x)

        cycle_len = round(self.fs / peak_freq)
        bpm = peak_freq * 60
        if bpm > 140 or bpm < 35:
            # self.valid_flag = False
            return True, cycle_len, bpm
        else:
            return False, cycle_len, bpm

    def get_cycle(self):
        cycle_list = []
        cycle_len_list = []
        lr_check_list = []
        if len(self.dbp_idx) >= 2:
            for i in range(len(self.dbp_idx) - 1):
                cycle = self.input_sig[self.dbp_idx[i]:self.dbp_idx[i + 1]]
                cycle_list.append(cycle)
                cycle_len_list.append(len(cycle))
                lr_check_list.append(abs(cycle[0] - cycle[-1]))
            avg_cycle_len = np.mean(cycle_len_list, dtype=np.int)

            peak_bpm = (self.fs / avg_cycle_len) * 60
            length_order = np.argsort(np.abs(np.array(cycle_len_list) - avg_cycle_len))
            diff_order = np.argsort(lr_check_list)
            total_order = length_order[np.where((diff_order == length_order) == True)]
            if len(total_order) > 0:
                most_promising_cycle_idx = total_order[0]
            else:
                most_promising_cycle_idx = length_order[0]
            # check if cycle lengths are similar
            if not self.get_mahalanobis_dis(cycle_len_list):
                return False, cycle_list[most_promising_cycle_idx]
            if 35 < peak_bpm < 140 or 35 < self.fft_bpm < 140:
                return True, cycle_list[most_promising_cycle_idx]
            else:
                return False, cycle_list[most_promising_cycle_idx]
        else:
            return False, np.zeros(1)
