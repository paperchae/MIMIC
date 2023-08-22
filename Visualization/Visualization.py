import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def bland_altman_plot(models: list, target_hr: np.ndarray, test_hr: np.ndarray,
                      title=None, xlabel=None, ylabel=None, show_plot=True):
    """

    :param test_hr:
    :param target_hr:
    :param models:
    :param title:
    :param xlabel:
    :param ylabel:
    :param show_plot:
    :return:
    """
    color = ['royalblue', 'mediumseagreen', 'darkorange', 'firebrick', 'darkviolet']

    mean = np.mean(np.vstack((target_hr, test_hr)), axis=0)
    difference = target_hr - test_hr
    bias = np.mean(difference)
    std = np.std(difference)
    lower_limit = bias - 1.96 * std
    upper_limit = bias + 1.96 * std
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=15, fontweight='bold')
    # for i in range(len(self.test)):
    plt.scatter(x=mean, y=difference, color=color[0], edgecolors='black', label=models[0])

    plt.axhline(y=bias, color='black', linestyle='--')
    plt.text((np.max(mean) - 12), (bias - 5), 'Mean: {}'.format(str(round(bias, 3))), color='black')
    plt.axhline(y=lower_limit, color='gray', linestyle='--')
    plt.text((np.max(mean) - 12), (lower_limit - 5), '-1.96SD: {}'.format(str(round(lower_limit, 3))),
             color='black')
    plt.axhline(y=upper_limit, color='gray', linestyle='--')
    plt.text((np.max(mean) - 12), (upper_limit - 5), '+1.96SD: {}'.format(str(round(upper_limit, 3))),
             color='black')
    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.legend(title='Model', loc='upper right', fontsize='small')
    plt.show()


def heatmap(df):
    """
    Draw heatmap of correlation matrix
    :param df: dataframe
    :return:
    """
    df = df.dropna()
    sns.set(style="whitegrid", color_codes=True)
    sns.pairplot(df)
    plt.figure(figsize=(10, 8))
    plt.title('Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(method='pearson'), vmax=1.0, square=True, cmap=plt.cm.PuBu,
                annot=True, annot_kws={"size": 10}, fmt='.2f',
                linewidths=0.2, linecolor='white')
    plt.show()
