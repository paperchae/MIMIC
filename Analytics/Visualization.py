import seaborn as sns
import matplotlib.pyplot as plt

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
