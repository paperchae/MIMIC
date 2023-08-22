import torch
from Utils.Signal.Handler import Cleansing


# import Handler
class FFT:
    def __init__(self, input_sig, fs=60.):
        self.input_sig = input_sig
        self.fs = fs

    def fft(self, dc_removal=True, plot=False):
        if dc_removal:
            self.input_sig = Cleansing(self.input_sig, cpu=True).dc_removal()
        amp = torch.abs(torch.fft.rfft(self.input_sig, dim=-1) * (2 / self.input_sig.shape[-1]))
        freq = torch.fft.rfftfreq(self.input_sig.shape[-1], 1 / self.fs)
        # return torch.abs(torch.fft.rfft(self.input_sig, dim=-1)), torch.fft.rfftfreq(self.input_sig.shape[-1], 1 / self.fs)
        if plot:
            import matplotlib.pyplot as plt
            plt.title('FFT')
            plt.plot(freq, amp)
            plt.grid(True)
            plt.show()
        return amp, freq

    def peak_freq(self):
        amp, freq = self.fft(dc_removal=True)
        return freq[torch.argmax(amp)]


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate test signal
    # FS = 125
    # T = 1 / FS
    # time_len = 10
    # time = np.linspace(0, time_len, FS * time_len)
    # amp = [2, 1, 0.5, 0.25]
    # freq = [10, 20, 30, 40]
    # test_list = []
    # for a, f in zip(amp, freq):
    #     if a == 0:
    #         continue
    #     test_list.append(a * np.sin(2 * np.pi * f * time))
    # test = torch.tensor(np.sum(test_list, axis=0))

    data = pd.read_csv('/home/paperc/PycharmProjects/MIMIC/MIMIC_Clinical_Database/III/result/pleth.csv').to_numpy()[:,
           1:]
    test = torch.tensor(data[0][:1200])
    time = np.linspace(0, 10, 1200)
    plt.title('Signal')
    plt.plot(time, test, label='raw ppg')
    plt.plot(time, Cleansing(test, True).dc_removal(), label='dc removed ppg')
    plt.legend()
    plt.show()
    plt.title('DC Removed FFT ( for cycle detection )')
    amp, freq = FFT(test, fs=60).fft(dc_removal=True)
    plt.plot(freq, amp)
    plt.grid(True)
    plt.show()

    plt.title('FFT')
    amp, freq = FFT(test, fs=60).fft(dc_removal=False)
    plt.plot(freq, amp)
    plt.grid(True)
    plt.show()

    print('test')
