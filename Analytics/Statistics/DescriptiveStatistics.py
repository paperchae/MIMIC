import numpy as np
import statistics as st
from scipy import stats
import matplotlib.pyplot as plt


class CentralTendency:
    def __init__(self, data):
        self.data = np.round(data)

    def mean(self):
        return st.mean(self.data)

    def median(self):
        return st.median(self.data)

    def mode(self):
        return st.mode(self.data)


class Dispersion:
    # After Central Tendency, calculate degree of dispersion of data
    def __init__(self, data):
        self.data = np.round(data)

    def var(self):
        return np.var(self.data)

    def std(self):
        return np.std(self.data)

    def range(self):
        return np.max(self.data) - np.min(self.data)

    def q1(self):
        return np.percentile(self.data, 25)

    def q3(self):
        return np.percentile(self.data, 75)

    def iqr(self):  # Inter quartile range
        return self.q3() - self.q1()

    def cv(self):  # Coefficient  variation, larger, more dispersion
        return np.round(self.std() / CentralTendency(self.data).mean() * 100, 2)


class Distribution:
    def __init__(self, data):
        self.data = data
        #  if normal distribution, skewness = 0 / negative: left tail / positive: right tail
        self.skewness = stats.skew(self.data)
        #  if normal distribution, kurtosis = 0 / negative: flat / positive: sharp
        self.kurtosis = stats.kurtosis(self.data)

    def standard_normal_distribution(self):
        mean = np.mean(self.data)
        std = np.std(self.data)
        if np.abs(mean - 0) < 1e-6 and np.abs(std - 1) < 1e-6:
            return True
        else:
            return False

    def plot(self):
        normal = np.random.normal(CentralTendency(self.data).mean(), Dispersion(self.data).std(), len(self.data))
        plt.hist(normal, bins=100, density=True, alpha=0.5, label='normal')
        plt.hist(self.data, bins=100, density=True, alpha=0.5, label='data')
        iqr = Outlier(self.data).iqr_test()
        std = Outlier(self.data).std_test()
        plt.hist(self.data[iqr]/len(self.data), bins=100, density=True, alpha=0.5, label='iqr')
        # plt.hist(self.data[std], bins=100, density=True, alpha=0.5, label='std')
        plt.legend(loc='upper right')
        plt.show()


class Outlier:
    def __init__(self, data):
        self.data = np.round(data)

    def iqr_test(self):
        iqr = Dispersion(self.data).iqr()
        lower_threshold = Dispersion(self.data).q1() - 1.5 * iqr
        upper_threshold = Dispersion(self.data).q3() + 1.5 * iqr
        return np.where((self.data < lower_threshold) | (self.data > upper_threshold))

    def std_test(self):
        std = Dispersion(self.data).std()
        mean = CentralTendency(self.data).mean()
        return np.where((self.data < mean - 3 * std) | (self.data > mean + 3 * std))


if __name__ == "__main__":
    test = np.random.randn(1990) * 20 + 80

    out_test = Outlier(test).iqr_test()
    print(out_test)
    std_test = Outlier(test).std_test()
    print(std_test)
    Distribution(test).plot()
