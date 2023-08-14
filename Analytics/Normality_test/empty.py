import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class Normality:
    def __init__(self, data):
        self.data = data

    def central_limit_theorem(self):
        if np.shape(self.data)[0] > 30:
            return True
        else:
            return False

    def shapiro_wilk(self):
        shapiro_wilk = stats.shapiro(self.data)
        print(shapiro_wilk)
        pass

    def kolmogorov_smirnov(self):
        kolmogorov_smirnov = stats.kstest(self.data, 'norm')
        print(kolmogorov_smirnov)
        pass

    def test(self):
        if self.central_limit_theorem():
            if len(self.data) < 2000:
                self.shapiro_wilk()
            else:
                self.kolmogorov_smirnov()
        else:
            print('The sample size is too small for CLT')

    def qq_plot(self):
        pass


def confidence_interval(data, confidence_level=0.95):
    """
    Calculate confidence interval of sampled data
    :param data:
    :param confidence:
    :return:
    """
    if not Normality(data).central_limit_theorem():
        raise ValueError('The sample size is too small for CLT')

    """
    confidence level means the probability of the mean of the samples from population
    is in the confidence interval
    """
    z_dict = {0.90: 1.65, 0.95: 1.96, 0.99: 2.58}
    z = z_dict[confidence_level]

    n = np.shape(data)[0]
    mean = np.mean(data)
    std = np.std(data)

    margin_of_error = z * std / np.sqrt(n)
    print(str(mean) + ' +- ' + str(margin_of_error))
    return mean, margin_of_error
