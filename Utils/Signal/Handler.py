import torch
import numpy as np


class Cleansing:
    def __init__(self, signals, cpu=False):
        self.signals = signals
        self.is_tensor = True if type(signals) == torch.Tensor else False
        if not self.is_tensor:
            self.signals = torch.from_numpy(self.signals.copy()).to(torch.float)
        else:
            self.signals = signals.to(torch.float)
        if cpu:
            print('!Currently accessing Cleansing Class with cpu, it might take a long time.')
            pass
        else:
            self.signals = self.signals.to('cuda')

    def detrend(self, Lambda=100):
        """
        * Only available with cuda device*
        This function applies a detrending filter to the 1D signals with linear trend.
        with diagonal matrix D, using torch batch matrix multiplication

        Based on the following article "An advanced detrending method with application
        to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

        ** It may take 10 GB of GPU memory for signals with 18000 length

        :param Lambda: Smoothing parameter
        :return: Detrended signals
        """
        if not torch.cuda.is_available():
            raise Exception('Cuda device is not available')
        # if not self.is_tensor:
        #     self.signals = torch.from_numpy(self.signals.copy())
        # signals = self.signals.to(torch.float).to('cuda')
        if self.signals.dim() == 1:
            self.signals = self.signals.unsqueeze(0)
        test_n, length = self.signals.shape

        H = torch.eye(length).to('cuda')
        ones = torch.diag(torch.ones(length - 2)).to('cuda')
        zeros_1 = torch.zeros((length - 2, 1)).to('cuda')
        zeros_2 = torch.zeros((length - 2, 2)).to('cuda')

        D = torch.cat((ones, zeros_2), dim=-1) + \
            torch.cat((zeros_1, -2 * ones, zeros_1), dim=-1) + \
            torch.cat((zeros_2, ones), dim=-1)

        detrended_signal = torch.bmm(self.signals.unsqueeze(1),
                                     (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).expand(test_n, -1,
                                                                                                       -1)).squeeze()

        if detrended_signal.dim() == 1:
            offset = torch.mean(self.signals, dim=-1)
        else:
            offset = torch.mean(self.signals, dim=-1, keepdim=True)
        detrended_signal += offset
        return detrended_signal

    def dc_removal(self):
        """
        Removes the dc value from the signals
        :return: dc removed signals
        """
        return self.signals - torch.mean(self.signals, dim=-1, keepdim=True)


class Manipulation:
    """
    Signal Manipulation class
    Supports to_chunks, down-Sample, normalize
    signals: torch.tensor(n, length)
    """

    def __init__(self, signals):
        self.signals = signals
        self.is_tensor = True if type(signals) == torch.Tensor else False
        self.data_type = type(signals)
        # self.device = signals.get_device() if type(signals) == torch.Tensor else None
        self.device = 'cuda' if type(signals) == torch.Tensor else 'cpu'
        self.n, self.length = signals.shape

    def to_chunks(self, chunk_size: int):
        """
        Split signals into chunks
        * if remainder exists, drop the last chunk for splitting
        :param chunk_size:
        :return: return signals in shape (n, -1, chunk_size)
        """
        if self.length < chunk_size:
            raise ValueError('Divider(Chunk size) is larger than signal length')

        return self.signals[:, :self.length - self.length % chunk_size].reshape(self.n, -1, chunk_size)

    def down_sample(self, from_fs: int = None, to_fs: int = None):
        """
        Down-sample signals from from_fs to to_fs
        :param from_fs: int, original sampling frequency
        :param to_fs: int, target sampling frequency
        :return: down sampled signals in shape (n, -1)
        """
        if self.data_type == np.ndarray:
            return np.array([x[0::from_fs // to_fs] for x in self.signals])
        elif self.data_type == torch.Tensor:
            return torch.stack([x[0::from_fs // to_fs] for x in self.signals])

    def normalize(self, mode='minmax'):
        """
        Normalize 1D signals
        :param mode: str, 'minmax' or 'zscore'
                     if 'minmax': normalize to [0, 1]
                     if 'zscore': normalize to 0 mean and 1 std
        :return: normalized signals
        """
        if self.data_type != torch.Tensor:
            self.signals = torch.from_numpy(self.signals.copy())
        if mode == 'minmax':
            min_val = torch.min(self.signals, dim=-1, keepdim=True)[0]
            max_val = torch.max(self.signals, dim=-1, keepdim=True)[0]
            return (self.signals - min_val) / (max_val - min_val)
        elif mode == 'zscore':
            if self.signals.dtype != torch.float:
                self.signals = self.signals.to(torch.float)
            mean = torch.mean(self.signals, dim=-1, keepdim=True)
            std = torch.std(self.signals, dim=-1, keepdim=True)
            return (self.signals - mean) / std

    def trim_mask(self):
        """
        Remove mask from signals
        :return: signals without mask
        """
        return self.signals[:, :torch.max(torch.sum(self.signals > 0, dim=-1))]

    def remove_negative(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    # Cleansing - Detrend
    test_sig = torch.tensor(
        pd.read_csv('../MIMIC_Clinical_Database/III/result/pleth.csv').to_numpy()[:, 1:][:2, 0:1000])

    # test_sig = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to('cuda')
    c = Cleansing(test_sig).detrend()
    plt.title('1D signal de-trend example')
    plt.plot(test_sig[0].cpu().numpy(), color='royalblue', label='original')
    plt.plot(c[0].cpu().numpy(), label='detrended', color='orange')
    plt.legend()
    plt.show()
    plt.close()

    # Manipulation - to_chunks
    chunks = Manipulation(test_sig).to_chunks(3)
    print('Original shape: ', test_sig.shape)
    print('Split shape:', chunks.shape)

    # Manipulation - down_sample
    down_sampled = Manipulation(test_sig).down_sample(2, 1)
    print('Down sampled shape:', test_sig.shape)
    print('Down sampled shape:', down_sampled.shape)

    # Manipulation - normalize
    normalized_1 = Manipulation(test_sig).normalize('minmax')
    normalized_2 = Manipulation(test_sig).normalize('zscore')
    plt.title('1D signal normalization example')
    plt.plot(test_sig[0].cpu().numpy(), label='original', color='royalblue')
    plt.plot(normalized_2[0].cpu().numpy(), label='zscore', color='green')
    plt.plot(normalized_1[0].cpu().numpy(), label='minmax', color='orange')
    plt.legend()
    plt.show()
    plt.close()
